use std::{cmp::max, collections::BTreeSet, hash::Hash, mem};

use anyhow::ensure;
use fxhash::FxHashMap;

use crate::{Decompose, Extract, MemSized};

/// A Dot is pair of the form (replica id, sequence number) that uniquely identifies an operation.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Dot<I>(pub I, pub u64);

/// A Dot Context is a set of dots represented in a compressed by two components: a vector clock
/// and a dot cloud. The vector clock component represents the contiguous portion of causal context
/// whereas the dot cloud admits history gaps.
///
/// Given the following set of Dots `s = {("A", 1), ("A", 2), ("A, 3"), ("B", 1), ("B", 4)}` then
/// the compressed dot context `ctx = { clock: [("A", 3), ("B", 1)], cloud: {("B", 4)} }`.
///
/// # Example
///
/// ```
/// use crdt::{Dot, DotContext};
///
/// let mut ctx = DotContext::new();
///
/// for i in 1..=2 {
///     ctx.next(&"a"); // Advance the history at replica "a".
///     assert_eq!(ctx.max(&"a"), i);
/// }
///
/// assert_eq!(ctx.contains(&Dot("a", 1)), true);
/// assert_eq!(ctx.contains(&Dot("a", 2)), true);
/// assert_eq!(ctx.contains(&Dot("a", 3)), false);
/// assert_eq!(ctx.contains(&Dot("b", 1)), false);
/// assert_eq!(ctx.contains(&Dot("b", 2)), false);
/// ```
#[derive(Clone, Debug, Default)]
pub struct DotContext<I> {
    clock: FxHashMap<I, u64>,
    cloud: BTreeSet<Dot<I>>,
}

/// The `Delta` type represents a view into the state of a given state. They can be joined with any
/// other [`DotContext`] in order to synchronize. They are read-only but can be easily converted
/// into a [`DotContext`] using the trait [`From`].
///
/// This `struct` is created upon a mutation of a [`DotContext`].
///
/// [`From`]: std::convert::From
///
/// # Tips
///
/// Deltas can be used when it is required to clone a given state.
///
/// ```
/// use crdt::{Decompose, DotContext};
///
/// let mut ctx = DotContext::new();
///
/// ctx.next(&"a");
/// ctx.next(&"b");
/// ctx.next(&"c");
///
/// let delta = ctx.as_delta();
///
/// let copy = DotContext::from(delta); // The state of set is cloned here!
/// assert_eq!(ctx, copy);
/// ```
#[derive(Clone, Debug)]
pub struct Delta<'a, I> {
    ctx: &'a DotContext<I>,
    clock: Vec<(&'a I, &'a u64)>,
    cloud: Vec<(&'a I, &'a u64)>,
}

impl<I> DotContext<I> {
    /// Creates an empty `DotContext`.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            clock: FxHashMap::default(),
            cloud: BTreeSet::new(),
        }
    }

    /// Returns `true` if the dot context contains no clock entries and no cloud dots.
    pub fn is_empty(&self) -> bool {
        self.clock.is_empty() && self.cloud.is_empty()
    }
}

impl<I> DotContext<I>
where
    I: Eq + Hash,
{
    /// Returns `true`` if `self` is compressed, i.e., each cloud dot is ahead by more than one
    /// from the clock timestamp.
    pub fn is_compressed(&self) -> bool {
        self.cloud
            .iter()
            .all(|Dot(i, n)| !self.clock.get(i).is_some_and(|clock| *n <= clock + 1))
    }

    /// Returns the largest timestamp for `id` contained in this `DotContext`. If `id` does not
    /// exist this method returns 0.
    #[inline]
    pub fn max(&self, id: &I) -> u64 {
        let lower_bound = self.clock.get(id).copied().unwrap_or_default();
        let upper_bound = self
            .cloud
            .iter()
            .filter_map(|Dot(i, n)| if i == id { Some(n) } else { None })
            .max()
            .copied()
            .unwrap_or_default();

        max(lower_bound, upper_bound)
    }
}

impl<I> DotContext<I>
where
    I: Ord + Eq + Hash,
{
    /// Returns `true` if the `DotContext` contains a dot. A dot is contained in a `DotContext` if
    /// it is coverred by the vector clock component or it is contained in the cloud component.
    ///
    /// # Example
    ///
    /// ```
    /// use crdt::{Dot, DotContext};
    ///
    /// let mut ctx = DotContext::new();
    /// ctx.next(&"a");
    /// ctx.next(&"a");
    ///
    /// assert_eq!(ctx.contains(&Dot(&"a", 1)), true); // Coverred by the clock.
    /// assert_eq!(ctx.contains(&Dot(&"a", 2)), true); // Coverred by the clock.
    /// assert_eq!(ctx.contains(&Dot(&"a", 3)), false); // No longer coverred by the clock.
    /// assert_eq!(ctx.contains(&Dot(&"b", 3)), false); // Not in the cloud dot.
    /// ```
    pub fn contains(&self, dot: &Dot<I>) -> bool {
        let Dot(i, n) = dot;
        self.clock.get(i).is_some_and(|clock| n <= clock) || self.cloud.contains(dot)
    }
}

impl<I> DotContext<I>
where
    I: Clone + Eq + Hash,
{
    /// Advances the history at a given replica `id` and returns a [`Delta`] containing the most
    /// recent clock entry for the replica `id`.
    ///
    /// # Note
    ///
    /// This function assumes that replicas know the order of their own operations and do not
    /// impersonate other replicas. Thus, the returned timestamp in the delta will be equivalent to
    /// the [`max`] value.
    ///
    /// [`max`]: DotContext::max
    pub fn next(&mut self, id: &I) -> Delta<'_, I> {
        match self.clock.get_mut(id) {
            Some(clock) => *clock += 1,
            None => {
                self.clock.insert(id.clone(), 1);
            }
        }

        let entry = self
            .clock
            .get_key_value(id)
            .expect("entry must exist at this point");

        Delta {
            ctx: self,
            clock: vec![entry],
            cloud: vec![],
        }
    }
}

impl<I> DotContext<I>
where
    I: Clone + Eq + Ord + Hash,
{
    /// Compresses the representation of the dot context.
    ///
    /// The algorithm iterates in sorted order thorugh the cloud of dots and determines if each dot
    /// is either already present in the clock or it is increments the clock. If either of these,
    /// the dot is moved from the cloud to the clock, otherwise it remains intact.
    pub fn compress(&mut self) {
        self.cloud.retain(|Dot(i, n)| match self.clock.get_mut(i) {
            Some(clock) if *n == (*clock + 1) => {
                *clock += 1;
                false
            }
            Some(clock) if n <= clock => false,
            _ => true, // The join handles the contiguous history
        })
    }
}

impl<I> PartialEq for DotContext<I>
where
    I: Eq + Hash,
{
    fn eq(&self, other: &Self) -> bool {
        // FIXME: The semantics here are a bit weird. Should the equality account for strict
        // equality between the clock and the cloud or ensure that both contexts encode the same
        // history even though some internal differences may exist, e.g., a dot in the cloud that
        // it is not the most recent.
        self.clock == other.clock && self.cloud == other.cloud
    }
}

impl<I> From<Delta<'_, I>> for DotContext<I>
where
    I: Clone + Eq + Ord + Hash,
{
    fn from(value: Delta<'_, I>) -> Self {
        Self {
            clock: value
                .clock
                .into_iter()
                .map(|(i, n)| (i.clone(), *n))
                .collect(),
            cloud: value
                .cloud
                .into_iter()
                .map(|(i, n)| Dot(i.clone(), *n))
                .collect(),
        }
    }
}

impl<I> Decompose for DotContext<I>
where
    I: Clone + Eq + Ord + Hash,
{
    type Decomposition<'a> = Delta<'a, I> where I: 'a;

    fn as_delta(&self) -> Self::Decomposition<'_> {
        Delta {
            ctx: self,
            clock: self.clock.iter().collect(),
            cloud: self.cloud.iter().map(|Dot(i, n)| (i, n)).collect(),
        }
    }

    fn split(&self) -> Vec<Self::Decomposition<'_>> {
        let clock = self.clock.iter().map(|entry| Delta {
            ctx: self,
            clock: vec![entry],
            cloud: vec![],
        });

        let cloud = self.cloud.iter().map(|Dot(i, n)| Delta {
            ctx: self,
            clock: vec![],
            cloud: vec![(i, n)],
        });

        clock.chain(cloud).collect()
    }

    fn join(&mut self, deltas: Vec<Self::Decomposition<'_>>) {
        deltas.into_iter().for_each(|Delta { clock, cloud, .. }| {
            clock
                .into_iter()
                .for_each(|(i, n)| match self.clock.get_mut(i) {
                    Some(clock) => *clock = max(*clock, *n),
                    None => {
                        self.clock.insert(i.clone(), *n);
                    }
                });

            let missing_dots = cloud
                .into_iter()
                .filter_map(|(i, n)| {
                    let dot = Dot(i.clone(), *n);
                    (!self.cloud.contains(&dot)).then_some(dot)
                })
                .collect::<Vec<_>>();

            self.cloud.extend(missing_dots);
        });

        self.compress()
    }

    fn difference<'a>(&'a self, remote: &'a Self) -> Self::Decomposition<'a> {
        let clocks = self
            .clock
            .iter()
            .filter(|(i, clock)| match remote.clock.get(i) {
                Some(n) => n < clock,
                None => true,
            });

        let dots = self.cloud.difference(&remote.cloud).map(|Dot(i, n)| (i, n));

        Delta {
            ctx: self,
            clock: clocks.collect(),
            cloud: dots.collect(),
        }
    }
}

/// A `DotKind` represents a type of entry that can be extracted from a [`Delta`] that is an
/// irredundant join-decomposition.
#[derive(Clone, Debug, Hash)]
pub enum DotKind<'a, I> {
    /// An entry in the vector clock component.
    Clock(&'a I, &'a u64),
    /// An entry contained in the dot cloud component.
    Cloud(&'a I, &'a u64),
}

impl<'a, I> Extract for Delta<'a, I>
where
    I: Hash,
{
    type Output = DotKind<'a, I>;

    fn extract(&self) -> anyhow::Result<Self::Output> {
        let values = self.clock.len() + self.cloud.len();
        ensure!(
            values == 1,
            "decomposition should contain a single value, but instead got {values} values"
        );

        match self.clock.first() {
            Some((i, n)) => Ok(DotKind::Clock(*i, n)),
            None => match self.cloud.first() {
                Some((i, n)) => Ok(DotKind::Cloud(*i, n)),
                None => unreachable!("decompositions contains at least a value"),
            },
        }
    }
}

impl MemSized for DotContext<String> {
    fn size_of(&self) -> usize {
        let ids = self
            .clock
            .keys()
            .chain(self.cloud.iter().map(|Dot(id, _)| id))
            .map(String::len)
            .sum::<usize>();

        ids + (self.cloud.len() + self.clock.len()) * mem::size_of::<u64>()
    }
}

impl<'a, I> Delta<'a, I> {
    /// Creates an empty `Delta` from a given `DotContext`.
    pub(crate) fn empty_with(ctx: &'a DotContext<I>) -> Self {
        Self {
            ctx,
            clock: vec![],
            cloud: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use fxhash::FxHashMap;

    use crate::{dot_context::DotKind, Decompose, Dot, DotContext, Extract, MemSized};

    #[test]
    fn emptiness_test() {
        let empty_ctx = DotContext::<()>::new();
        assert!(empty_ctx.is_empty());
        assert!(empty_ctx.is_compressed());
    }

    #[test]
    fn membership_test() {
        let ctx = DotContext {
            clock: FxHashMap::from_iter([("a", 2), ("b", 3)]),
            cloud: BTreeSet::from([Dot("a", 3), Dot("c", 3)]),
        };

        assert!(!ctx.is_empty());

        for i in 1..=3 {
            assert_eq!(ctx.contains(&Dot("a", i)), true);
            assert_eq!(ctx.contains(&Dot("b", i)), true);
        }

        for i in 4..=5 {
            assert_eq!(ctx.contains(&Dot("a", i)), false);
            assert_eq!(ctx.contains(&Dot("b", i)), false);
        }

        assert_eq!(ctx.contains(&Dot("c", 2)), false);
        assert_eq!(ctx.contains(&Dot("c", 3)), true);
        assert_eq!(ctx.contains(&Dot("c", 4)), false);
    }

    #[test]
    fn max_and_next_test() {
        let mut ctx = DotContext {
            clock: FxHashMap::from_iter([("a", 2), ("b", 3), ("d", 3)]),
            cloud: BTreeSet::from([Dot("a", 3), Dot("b", 2), Dot("c", 3)]),
        };

        let expected = [
            ("a", 3, 3),
            ("b", 4, 4),
            ("c", 1, 3),
            ("d", 4, 4),
            ("e", 1, 1),
        ];
        for (i, n, m) in expected {
            let delta = ctx.next(&i);
            let ctx_from_delta = DotContext::from(delta);
            assert_eq!(ctx_from_delta.max(&i), n);
            assert_eq!(ctx.max(&i), m);
        }
    }

    #[test]
    fn compression_test() {
        let mut ctx = DotContext {
            clock: FxHashMap::from_iter([("a", 2), ("b", 3)]),
            cloud: BTreeSet::from([
                Dot("a", 1),
                Dot("a", 2),
                Dot("a", 3),
                Dot("b", 5),
                Dot("c", 3),
            ]),
        };

        assert!(!ctx.is_compressed());

        ctx.compress();
        assert!(ctx.is_compressed());

        let expected_ctx = DotContext {
            clock: FxHashMap::from_iter([("a", 3), ("b", 3)]),
            cloud: BTreeSet::from([Dot("b", 5), Dot("c", 3)]),
        };

        assert_eq!(ctx, expected_ctx);
    }

    #[test]
    fn irredudant_join_decomposition_test() {
        let mut ctx = DotContext {
            clock: FxHashMap::from_iter([("a", 2), ("b", 3)]),
            cloud: BTreeSet::from([Dot("a", 3), Dot("b", 5), Dot("c", 3)]),
        };

        assert!(!ctx.is_compressed());

        let irredundant_join_decomposition = ctx.split();
        assert_eq!(irredundant_join_decomposition.len(), 5);

        let single_items = irredundant_join_decomposition
            .iter()
            .all(|d| d.clock.len() + d.cloud.len() == 1);
        assert!(single_items);

        let mut cloned_ctx = DotContext::new();
        cloned_ctx.join(irredundant_join_decomposition);

        // Compression happens when joining
        assert!(cloned_ctx.is_compressed());
        assert_ne!(ctx, cloned_ctx);

        ctx.compress();
        assert_eq!(ctx, cloned_ctx);
    }

    #[test]
    fn difference_test() {
        let local = DotContext {
            clock: FxHashMap::from_iter([("a", 2), ("b", 3), ("c", 3), ("d", 9)]),
            cloud: BTreeSet::from([Dot("a", 3), Dot("b", 5), Dot("c", 3)]),
        };

        let remote = DotContext {
            clock: FxHashMap::from_iter([("a", 2), ("b", 2), ("c", 4), ("e", 5)]),
            cloud: BTreeSet::from([Dot("a", 3), Dot("b", 5), Dot("c", 2)]),
        };

        let diff = DotContext::from(local.difference(&remote));
        let expected_diff = DotContext {
            clock: FxHashMap::from_iter([("b", 3), ("d", 9)]),
            cloud: BTreeSet::from([Dot("c", 3)]),
        };
        assert_eq!(diff, expected_diff);

        let local = DotContext {
            clock: FxHashMap::from_iter([("a", 2), ("b", 3)]),
            cloud: BTreeSet::from([Dot("a", 3), Dot("b", 5)]),
        };

        let remote = DotContext {
            clock: FxHashMap::from_iter([("a", 2), ("b", 3), ("c", 3), ("e", 5)]),
            cloud: BTreeSet::from([Dot("a", 3), Dot("b", 5), Dot("c", 3)]),
        };

        let empty_diff = DotContext::from(local.difference(&remote));
        assert!(empty_diff.is_empty());
    }

    #[test]
    fn extraction_test() {
        let empty_ctx = DotContext::<()>::new();
        let empty_delta = empty_ctx.as_delta();

        let extraction = empty_delta.extract();
        assert!(
            extraction.is_err(),
            "extraction is working with empty deltas"
        );

        let ctx = DotContext {
            clock: FxHashMap::from_iter([("a", 3)]),
            cloud: BTreeSet::new(),
        };
        let clock_delta = ctx.as_delta();

        let extraction = clock_delta.extract();
        assert!(matches!(extraction, Ok(DotKind::Clock(&"a", 3))));

        let ctx = DotContext {
            clock: FxHashMap::default(),
            cloud: BTreeSet::from([Dot("a", 3)]),
        };
        let cloud_delta = ctx.as_delta();

        let extraction = cloud_delta.extract();
        assert!(matches!(extraction, Ok(DotKind::Cloud(&"a", 3))));

        let ctx = DotContext {
            clock: FxHashMap::from_iter([("a", 3)]),
            cloud: BTreeSet::from([Dot("a", 3)]),
        };
        let large_delta = ctx.as_delta();

        let extraction = large_delta.extract();
        assert!(
            extraction.is_err(),
            "extraction is working with large deltas"
        );
    }

    #[test]
    fn size_of_test() {
        let emtpy_ctx = DotContext::new();
        assert_eq!(emtpy_ctx.size_of(), 0);

        let ctx = DotContext {
            clock: FxHashMap::from_iter([(String::from("a"), 3), (String::from("zz"), 3)]),
            cloud: BTreeSet::from([Dot(String::from("a"), 5), Dot(String::from("b"), 5)]),
        };
        assert_eq!(ctx.size_of(), 1 + 8 + 2 + 8 + 1 + 8 + 1 + 8);
    }
}
