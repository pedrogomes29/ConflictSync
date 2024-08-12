#![allow(dead_code)]

mod dot_context;
mod gcounter;
mod gset;
mod orset;

pub use crate::dot_context::{Dot, DotContext};
pub use crate::gcounter::GCounter;
pub use crate::gset::GSet;

use std::hash::Hash;

/// The `Decompose` trait allows data types to support not only deltas but irredundant
/// join-decomposditions. This trait provides a way for clients to use these in the way that they
/// wish.
pub trait Decompose {
    type Decomposition<'a>
    where
        Self: 'a;

    /// Provides the only irredundant join-decompositions possible over the `self`.
    ///
    /// The implementation depends on the data type, more specifically, on the distributive
    /// join-semilattice that models the state of a given type.
    ///
    /// # Tips
    ///
    /// Determining the number the cardinality of a irredundant join-decomposition can be useful in
    /// scenarios of synchronization. However, only for [`GSet`]'s this value can be computed from
    /// the data type itself. For other data types (and also for [`GSet`]'s) use the following.
    ///
    /// ```
    /// use crdt::{Decompose, GSet};
    ///
    /// let mut set = GSet::default();
    /// set.insert("a");
    /// set.insert("b");
    ///
    /// assert_eq!(set.len(), 2);
    /// assert_eq!(set.len(), set.split().len());
    /// ```
    ///
    /// [`GSet`]: gset::GSet
    fn split(&self) -> Vec<Self::Decomposition<'_>>;

    /// Allows to join several deltas and join them together with `self`.
    fn join(&mut self, deltas: Vec<Self::Decomposition<'_>>);

    /// Computes the difference between two different states `self` and `remote`. In essence, it
    /// returns the portion of state present at `self` that does not exist in `remote`.
    ///
    /// Each data type provides its own implementations as this method depends on the irredundant
    /// join-decompositions of `self` and `remote`. This function represents the function $\Delta$
    /// first described in this [paper](https://arxiv.org/pdf/1803.02750).
    fn difference<'a>(&'a self, remote: &'a Self) -> Self::Decomposition<'a>;
}

/// The `Extract` trait allows to extract single values given a `Decomposition`. If such a
/// `Decomposition` is empty or contains more than one item, an error is returned back to the
/// caller.
///
/// Notice that it imposes a trait bound on the type T, which represents the output type for the
/// scenario where the extraction succeds. The values extracted are intended to be hashed to enable
/// efficient digest-driven state-based CRDT synchronization.
pub trait Extract<T>
where
    T: Hash,
{
    type Decomposition<'a>
    where
        Self: 'a;

    fn extract(delta: &Self::Decomposition<'_>) -> anyhow::Result<T>;
}
