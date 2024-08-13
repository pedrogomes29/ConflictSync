#![allow(dead_code)]

mod awset;
mod dot_context;
mod gcounter;
mod gset;

pub use crate::dot_context::{Dot, DotContext};
pub use crate::gcounter::GCounter;
pub use crate::gset::GSet;

use std::hash::Hash;
use std::mem;

/// The `Decompose` trait allows data types to support not only deltas but irredundant
/// join-decomposditions. This trait provides a way for clients to use these in the way that they
/// wish.
pub trait Decompose {
    type Decomposition<'a>
    where
        Self: 'a;

    /// Extracts a `Delta` containing the entire `Self` state.
    fn as_delta(&self) -> Self::Decomposition<'_>;

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
    /// join-decompositions of `self` and `remote`. This function represents the function `Delta`
    /// first described in this [paper](https://arxiv.org/pdf/1803.02750).
    fn difference<'a>(&'a self, remote: &'a Self) -> Self::Decomposition<'a>;
}

/// The `Extract` trait allows to extract single values given a `Decomposition`. If such a
/// `Decomposition` is empty or contains more than one item, an error is returned back to the
/// caller.
///
/// Notice that it imposes a trait bound on the associated type `Output`. The values extracted are
/// intended to be hashed to enable efficient digest-driven state-based CRDT synchronization.
pub trait Extract {
    /// The resulting type returned by `extract` if successful.
    type Output: Hash;

    /// Extracts an hashable type from a irredudant join-decomposition.
    fn extract(&self) -> anyhow::Result<Self::Output>;
}

/// The `MemSized` trait is an helper that allows to obtain the memory expenditure of a particular
/// data type.
///
/// Depending on the kind of values held by different data types, one may need to implement a
/// custom `size_of` function. In such scenarios, this trait can be implemented for the needed
/// value type.
pub trait MemSized {
    /// Returns the size in bytes of `self`.
    fn size_of(&self) -> usize;
}

impl MemSized for String {
    fn size_of(&self) -> usize {
        self.len()
    }
}

macro_rules! impl_mem_sized_for_nums {
    ($($t:ty),*) => {
        $(
            impl MemSized for $t {
                fn size_of(&self) -> usize {
                    mem::size_of::<$t>()
                }
            }
        )*
    };
}

impl_mem_sized_for_nums!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64);
