use std::{
    borrow::Borrow,
    collections::hash_set::{IntoIter, Iter},
    hash::Hash,
};

use anyhow::{ensure, Ok};
use fxhash::FxHashSet;

use crate::{Decompose, Extract};

/// A GSet is a grow-only state and a state-based CRDTs, arguably, the simplest of them all.
/// As its name suggests, this data type only supports insertion and membership querying.
///
/// # Implementation
///
/// The implementation of the [`GSet`] wraps a [`HashSet`] from the standard library and restricts
/// the interface to contain the appropriate methods.
///
/// [`HashSet`]: std::collections::HashSet
///
/// # Example
///
/// ```
/// use crdt::GSet;
///
/// let mut set = GSet::new();
///
/// // Once inserted an element can never be deleted!
/// set.insert("a");
/// set.insert("b");
///
/// if set.contains("a") {
///     println!("The GSet contains the letter a");
/// }
/// ```
#[derive(Clone, Debug, Default)]
pub struct GSet<T> {
    inner: FxHashSet<T>,
}

/// The `Delta` type represents a view into the state of a given state. They can be joined with any
/// other [`GSet`] in order to synchronize. They are read-only but can be easily converted into a
/// [`GSet`] using the trait [`From`].
///
/// [`From`]: std::convert::From
///
/// # Tips
///
/// [`Delta`] can be used when it is required to clone a given state.
///
/// ```
/// use crdt::GSet;
///
/// let mut set = GSet::new();
///
/// set.insert("a");
/// set.insert("b");
/// set.insert("c");
///
/// let delta = set.as_delta();
///
/// let copy = GSet::from(delta);   // The state of set is cloned here!
/// assert_eq!(set, copy);
/// ```
#[derive(Clone)]
pub struct Delta<'a, T> {
    set: &'a GSet<T>,
    elems: Vec<&'a T>,
}

impl<T> GSet<T> {
    /// Creates an empty [`GSet`], i.e., withtout any values.
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
    /// use crdt::GSet;
    ///
    /// // Create an empty set
    /// let set: GSet<i32> = GSet::new();
    ///
    /// assert!(set.is_empty());
    /// assert_eq!(set.len(), 0);
    /// ```
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: FxHashSet::default(),
        }
    }
}

impl<T> GSet<T> {
    /// An iterator visiting all the elements in arbitrary order.
    /// Since this is a wrapper around [`HashSet`] the iterator returned is the internal set iterator.
    ///
    /// [`HashSet`]: std::collections::HashSet
    ///
    /// # Examples
    ///
    /// ```
    /// use crdt::GSet;
    ///
    /// let mut set = GSet::new();
    /// set.insert("a");
    /// set.insert("b");
    ///
    /// for x in set.iter() {
    ///     println!("{x}");
    /// }
    /// ```
    pub fn iter(&self) -> Iter<'_, T> {
        self.inner.iter()
    }

    /// Returns `true` if the set contains no elements.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the number of elements in the set, i.e., its cardinality.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Transforms the `self` into a `Delta` object that contains its entire state.
    pub fn as_delta(&self) -> Delta<'_, T> {
        Delta {
            set: self,
            elems: self.inner.iter().collect(),
        }
    }
}

impl<T> GSet<T>
where
    T: Eq + Hash,
{
    /// Returns `true` if the set contains a value.
    pub fn contains<Q: ?Sized + Hash + Eq>(&self, value: &Q) -> bool
    where
        T: Borrow<Q>,
    {
        self.inner.contains(value)
    }

    /// Returns `true` if `self` and `other` are disjoint, i.e., `self` does not contain any
    /// values from `other` and vice-versa.
    pub fn is_disjoint(&self, other: &GSet<T>) -> bool {
        self.inner.is_disjoint(&other.inner)
    }

    /// Returns `true` if `self` is a subset of `other`, i.e., all the values of `self` are
    /// contained in `other`.
    pub fn is_subset(&self, other: &GSet<T>) -> bool {
        self.inner.is_subset(&other.inner)
    }

    /// Returns `true` if `self` is a superset of `other`, i.e., `self` contains at least all the
    /// values of `other`.
    pub fn is_superset(&self, other: &GSet<T>) -> bool {
        self.inner.is_superset(&other.inner)
    }

    /// Adds a value to the set.
    /// It creates `Some` delta if the value is inserted in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// use crdt::GSet;
    ///
    /// let mut set = GSet::new();
    ///
    /// assert!(set.insert("a").is_some());
    /// assert!(set.insert("b").is_some());
    /// assert!(set.insert("b").is_none());
    /// assert_eq!(set.len(), 2);
    /// ```
    pub fn insert(&mut self, value: T) -> Option<Delta<'_, T>>
    where
        T: Clone,
    {
        // FIXME: Change this when `get_or_insert` becomes stable. This way it would be possile to
        // remove the need for cloning value. A workaround would be to change the implementation to
        // use a HashMap, but it feels cumbersome and makes everything more complex.
        // See more: https://github.com/rust-lang/rust/pull/60894
        if self.inner.contains(&value) {
            return None;
        }

        self.inner.insert(value.clone());
        self.inner.get(&value).map(|v| Delta {
            set: self,
            elems: vec![v],
        })
    }
}

impl<T> PartialEq for GSet<T>
where
    T: Eq + Hash,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<T> From<Delta<'_, T>> for GSet<T>
where
    T: Clone + Eq + Hash,
{
    fn from(value: Delta<'_, T>) -> Self {
        Self {
            inner: FxHashSet::from_iter(value.elems.into_iter().cloned()),
        }
    }
}

impl<T> IntoIterator for GSet<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

impl<T> Decompose<T> for GSet<T>
where
    T: Eq + Hash,
{
    type Decomposition<'a> = Delta<'a, T> where T: 'a;

    fn split(&self) -> Vec<Self::Decomposition<'_>> {
        self.iter()
            .map(|v| Delta {
                set: self,
                elems: vec![v],
            })
            .collect()
    }

    fn join(&mut self, deltas: Vec<Self::Decomposition<'_>>)
    where
        T: Clone,
    {
        let unknown_elements = deltas
            .into_iter()
            .flat_map(|d| d.elems)
            .filter(|v| !self.inner.contains(v))
            .cloned()
            .collect::<Vec<_>>();

        self.inner.extend(unknown_elements);
    }

    fn difference<'a>(&'a self, remote: &'a Self) -> Self::Decomposition<'a> {
        Delta {
            set: self,
            elems: self.inner.difference(&remote.inner).collect(),
        }
    }
}

impl<'b, T> Extract<&'b T> for GSet<T>
where
    T: Hash,
{
    type Decomposition<'a> = Delta<'b, T> where T: 'a;

    fn extract(delta: &Self::Decomposition<'b>) -> anyhow::Result<&'b T> {
        let values = delta.elems.len();
        ensure!(
            values == 1,
            "decomposition should contain a single value, but instead got {values} values"
        );

        match delta.elems.first() {
            Some(value) => Ok(value),
            None => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use fxhash::FxHashSet;

    use crate::{Decompose, Extract, GSet};

    #[test]
    fn insertion_and_membership_test() {
        let mut set = GSet::new();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);

        assert!(set.insert("a").is_some_and(|d| d.elems.len() == 1));
        assert!(set.insert("b").is_some_and(|d| d.elems.len() == 1));
        assert!(set.insert("b").is_none());

        assert!(!set.is_empty(), "set shouldn't be empty");
        assert_eq!(set.len(), 2, "set should have only 2 elements");

        let elems = ["a", "b"];
        for x in elems {
            assert!(set.contains(x), "set does not contain element {x}");
        }
    }

    #[test]
    fn irredudant_join_decomposition_test() {
        let mut set = GSet::new();

        set.insert("a");
        set.insert("b");
        set.insert("c");

        let irredudant_join_decomposition = set.split();
        assert_eq!(set.len(), irredudant_join_decomposition.len());

        // Check if all the generated deltas have a single value
        assert!(irredudant_join_decomposition
            .iter()
            .all(|d| d.elems.len() == 1));

        let mut remote = GSet::new();
        remote.join(irredudant_join_decomposition);

        assert_eq!(set, remote);
    }

    #[test]
    fn difference_test() {
        let mut local = GSet {
            inner: FxHashSet::from_iter(["a", "b", "c", "e"]),
        };

        let mut remote = GSet {
            inner: FxHashSet::from_iter(["a", "b", "d", "f"]),
        };

        let actual_local_diff = GSet::from(local.difference(&remote));
        let expected_local_diff = GSet {
            inner: FxHashSet::from_iter(["c", "e"]),
        };
        assert_eq!(actual_local_diff, expected_local_diff);

        let actual_remote_diff = GSet::from(remote.difference(&local));
        let expected_remote_diff = GSet {
            inner: FxHashSet::from_iter(["d", "f"]),
        };
        assert_eq!(actual_remote_diff, expected_remote_diff);

        local.join(vec![actual_remote_diff.as_delta()]);
        remote.join(vec![actual_local_diff.as_delta()]);
        assert_eq!(local, remote);

        let local_diff = GSet::from(local.difference(&remote));
        assert!(
            local_diff.is_empty(),
            "difference between equal sets has items"
        );

        let remote_diff = GSet::from(remote.difference(&local));
        assert!(
            remote_diff.is_empty(),
            "difference between equal sets has items"
        );
    }

    #[test]
    fn extraction_test() {
        let mut set = GSet::new();

        let empty_delta = set.as_delta();
        let extraction = GSet::extract(&empty_delta);
        assert!(
            extraction.is_err(),
            "extraction is working with empty deltas"
        );

        let delta = set
            .insert("a")
            .expect("insertion of a new element should produce `Some`");
        let extraction = GSet::extract(&delta);
        let expected = delta
            .elems
            .first()
            .expect("expected should contain at least one element");
        assert!(extraction.is_ok_and(|v| v == *expected));

        set.insert("b");
        let large_delta = set.as_delta();
        let extraction = GSet::extract(&large_delta);
        assert!(
            extraction.is_err(),
            "extraction is working with large deltas"
        );
    }
}
