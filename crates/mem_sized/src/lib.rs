use std::mem;

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

macro_rules! impl_mem_sized_for_strs {
    ($($t:ty),*) => {
        $(
            impl MemSized for $t {
                fn size_of(&self) -> usize {
                    self.len()
                }
            }

        )*
    };
}

impl_mem_sized_for_nums!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64);
impl_mem_sized_for_strs!(String, str);
