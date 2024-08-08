use std::{borrow::Borrow, cmp::max, hash::Hash};

use anyhow::{ensure, Ok};
use fxhash::FxHashMap;

use crate::{Decompose, Extract};

/// A GCounter is a grow-only counter and a state-based CRDTs. THis data type only supports the
/// increment and count operations. This is also a named data type meaning that replicas who share
/// this data type must be uniquely identified.
///
/// # Implementation
///
/// The implementation of a GCounter wraps a [`HashMap`] from the standard library. The replica ids
/// are the keys and the number of increments represent the keys.
///
/// [`HashMap`]: std::collections::HashMap
///
/// # Example
///
/// ```
/// use crdt::GCounter;
///
/// let mut counter = GCounter::new();
///
/// // Once an increment happens the counter increases for the remaining of its lifetime!
/// counter.increment(&"a");
/// counter.increment(&"b");
/// counter.increment(&"a");
///
/// if counter.count() == 3 {
///     println!("The GCounter was incremented 3 times");
/// }
/// ```
#[derive(Clone, Debug, Default)]
pub struct GCounter<I> {
    inner: FxHashMap<I, u64>,
}

/// The `Delta` type represents a view into the state of a given state. They can be joined with any
/// other [`GCounter`] in order to synchronize. They are read-only but can be easily converted into a
/// [`GCounter`] using the trait [`From`].
///
/// [`From`]: std::convert::From
///
/// # Tips
///
/// [`Delta`] can be used when it is required to clone a given state.
///
/// ```
/// use crdt::GCounter;
///
/// let mut counter = GCounter::new();
///
/// counter.increment(&"a");
/// counter.increment(&"b");
/// counter.increment(&"a");
///
/// let delta = counter.as_delta();
///
/// let copy = GCounter::from(delta);   // The state of counter is cloned here!
/// assert_eq!(counter, copy);
/// ```
#[derive(Clone)]
pub struct Delta<'a, I> {
    counter: &'a GCounter<I>,
    elems: Vec<(&'a I, &'a u64)>,
}

impl<I> GCounter<I> {
    /// Creates a [`GCounter`] set to the value of 0.
    ///
    /// # Performance
    ///
    /// For performance reasons, this implementations used [`fxhash`] which is faster than the
    /// SipHash 1-3 algorithm used by the standard library. Even though, it does not provide
    /// cryptographic security again DDoS hash attacks, the fact is that this is still a toy
    /// project.
    ///
    /// [`fxhash`]: fxhash
    ///
    /// # Example
    ///
    /// ```
    /// use crdt::GCounter;
    ///
    /// // Create an empty set
    /// let counter: GCounter<i32> = GCounter::new();
    /// assert_eq!(counter.count(), 0);
    /// ```
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: FxHashMap::default(),
        }
    }
}

impl<I> GCounter<I>
where
    I: Eq + Hash,
{
    /// Returns the count of the counter, i.e., the number of increments.
    ///
    /// # Example
    ///
    /// ```
    /// use crdt::GCounter;
    ///
    /// let mut counter = GCounter::new();
    /// assert_eq!(counter.count(), 0);
    ///
    /// counter.increment(&"a");
    /// counter.increment(&"a");
    /// assert_eq!(counter.count(), 2);
    /// ```
    pub fn count(&self) -> u64 {
        self.inner.values().sum()
    }

    /// Returns the count, i.e., the number of increments of a given `id`.
    ///
    /// # Example
    ///
    /// ```
    /// use crdt::GCounter;
    ///
    /// let mut counter = GCounter::new();
    /// assert_eq!(counter.count_of(&"a"), None);
    ///
    /// counter.increment(&"a");
    /// counter.increment(&"a");
    /// assert_eq!(counter.count_of(&"a"), Some(2));
    /// ```
    pub fn count_of<Q: ?Sized + Hash + Eq>(&self, id: &Q) -> Option<u64>
    where
        I: Borrow<Q>,
    {
        self.inner.get(id).copied()
    }

    /// Transforms the `self` into a `Delta` object that contains its entire state.
    pub fn as_delta(&self) -> Delta<'_, I> {
        Delta {
            counter: self,
            elems: self.inner.iter().collect(),
        }
    }
}

impl<I> GCounter<I>
where
    I: Clone + Eq + Hash,
{
    /// Increments an `id` and returns a [`Delta`] that contains the `id` and its corresponding
    /// counter. If the `id` is not present in the counter, a new entry is initialized with 1.
    ///
    /// # Example
    ///
    /// ```
    /// use crdt::GCounter;
    ///
    /// let mut counter = GCounter::new();
    /// assert_eq!(counter.count_of(&"a"), None);
    ///
    /// counter.increment(&"a");
    /// counter.increment(&"a");
    /// assert_eq!(counter.count_of(&"a"), Some(2));
    /// ```
    pub fn increment(&mut self, id: &I) -> Delta<'_, I> {
        self.add(id, 1)
    }

    /// Adds `value` to an `id` and returns a [`Delta`] that contains the `id` and its corresponding
    /// counter. If the `id` is not present in the counter, a new entry is initialized with `value`.
    ///
    /// # Example
    ///
    /// ```
    /// use crdt::GCounter;
    ///
    /// let mut counter = GCounter::new();
    /// assert_eq!(counter.count_of(&"a"), None);
    ///
    /// counter.add(&"a", 42);
    /// assert_eq!(counter.count_of(&"a"), Some(42));
    /// ```
    pub fn add(&mut self, id: &I, value: u64) -> Delta<'_, I> {
        match self.inner.get_mut(id) {
            Some(count) => *count += value,
            None => {
                self.inner.insert(id.clone(), value);
            }
        }

        let entry = self
            .inner
            .get_key_value(id)
            .expect("key not found in counter");
        Delta {
            counter: self,
            elems: vec![entry],
        }
    }
}

impl<I> PartialEq for GCounter<I>
where
    I: Eq + Hash,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<I> From<Delta<'_, I>> for GCounter<I>
where
    I: Clone + Eq + Hash,
{
    fn from(value: Delta<'_, I>) -> Self {
        Self {
            inner: FxHashMap::from_iter(value.elems.into_iter().map(|(id, v)| (id.clone(), *v))),
        }
    }
}

impl<I> Decompose<I> for GCounter<I>
where
    I: Eq + Hash,
{
    type Decomposition<'a> = Delta<'a, I> where I: 'a;

    fn split(&self) -> Vec<Self::Decomposition<'_>> {
        self.inner
            .iter()
            .map(|entry| Delta {
                counter: self,
                elems: vec![entry],
            })
            .collect()
    }

    fn join(&mut self, deltas: Vec<Self::Decomposition<'_>>)
    where
        I: Clone,
    {
        deltas
            .into_iter()
            .flat_map(|d| d.elems)
            .for_each(|(id, remote_value)| {
                match self.inner.get_mut(id) {
                    Some(local_value) => *local_value = max(*local_value, *remote_value),
                    None => {
                        self.inner.insert(id.clone(), *remote_value);
                    }
                };
            })
    }

    fn difference<'a>(&'a self, remote: &'a Self) -> Self::Decomposition<'a> {
        Delta {
            counter: self,
            elems: self
                .inner
                .iter()
                .filter(|(id, v)| match remote.inner.get(id) {
                    Some(value) => *v > value,
                    None => true,
                })
                .collect(),
        }
    }
}

impl<'b, I> Extract<(&'b I, &'b u64)> for GCounter<I>
where
    I: Hash,
{
    type Decomposition<'a> = Delta<'b, I> where I: 'a;

    fn extract(delta: &Self::Decomposition<'b>) -> anyhow::Result<(&'b I, &'b u64)> {
        let values = delta.elems.len();
        ensure!(
            values == 1,
            "decomposition should contain a single value, but instead got {values} values"
        );

        match delta.elems.first() {
            Some(value) => Ok(*value),
            None => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use fxhash::FxHashMap;

    use crate::{Decompose, Extract, GCounter};

    #[test]
    fn addition_and_counting_test() {
        let mut counter = GCounter::new();
        assert_eq!(counter.count(), 0, "empty counter different than 0");

        counter.increment(&"a");
        counter.increment(&"b");
        counter.increment(&"a");

        assert_eq!(counter.count(), 3);
        assert_eq!(counter.count_of(&"a"), Some(2));
        assert_eq!(counter.count_of(&"b"), Some(1));
        assert_eq!(counter.count_of(&"c"), None);

        counter.add(&"c", 42);
        assert_eq!(counter.count_of(&"c"), Some(42));
        assert_eq!(counter.count(), 45);
    }

    #[test]
    fn irredudant_join_decomposition_test() {
        let mut counter = GCounter::new();

        counter.increment(&"a");
        counter.increment(&"b");
        counter.increment(&"a");

        let irredundant_join_decomposition = counter.split();
        assert_eq!(irredundant_join_decomposition.len(), 2);

        // Check if all the generated deltas have a single value
        assert!(irredundant_join_decomposition
            .iter()
            .all(|d| d.elems.len() == 1));

        let mut remote = GCounter::new();
        remote.join(irredundant_join_decomposition);

        assert_eq!(counter, remote);
    }

    #[test]
    fn difference_test() {
        let mut local = GCounter {
            inner: FxHashMap::from_iter([("a", 2), ("b", 3), ("c", 1), ("e", 1)]),
        };

        let mut remote = GCounter {
            inner: FxHashMap::from_iter([("a", 2), ("b", 1), ("d", 1), ("e", 3)]),
        };

        let actual_local_diff = GCounter::from(local.difference(&remote));
        let expected_local_diff = GCounter {
            inner: FxHashMap::from_iter([("b", 3), ("c", 1)]),
        };
        assert_eq!(actual_local_diff, expected_local_diff);

        let actual_remote_diff = GCounter::from(remote.difference(&local));
        let expected_remote_diff = GCounter {
            inner: FxHashMap::from_iter([("d", 1), ("e", 3)]),
        };
        assert_eq!(actual_remote_diff, expected_remote_diff);

        local.join(vec![actual_remote_diff.as_delta()]);
        remote.join(vec![actual_local_diff.as_delta()]);
        assert_eq!(local, remote);

        let local_diff = GCounter::from(local.difference(&remote));
        assert_eq!(
            local_diff.count(),
            0,
            "difference between equal counters different than 0"
        );

        let remote_diff = GCounter::from(remote.difference(&local));
        assert_eq!(
            remote_diff.count(),
            0,
            "difference between equal counters different than 0"
        );
    }

    #[test]
    fn extraction_test() {
        let mut counter = GCounter::new();

        let empty_delta = counter.as_delta();
        let extraction = GCounter::extract(&empty_delta);
        assert!(
            extraction.is_err(),
            "extraction is working with empty deltas"
        );

        let delta = counter.increment(&"a");
        let extraction = GCounter::extract(&delta);
        let expected = delta
            .elems
            .first()
            .expect("expected should contain at least one element");
        assert!(extraction.is_ok_and(|v| v == *expected));

        counter.increment(&"b");
        let large_delta = counter.as_delta();
        let extraction = GCounter::extract(&large_delta);
        assert!(
            extraction.is_err(),
            "extraction is working with large deltas"
        );
    }
}
