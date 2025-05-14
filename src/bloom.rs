use std::{
    cmp::max,
    f64::consts::LN_2,
    hash::{BuildHasher, Hash, RandomState},
    marker::PhantomData,
};

use bitvec::{bitvec, slice::BitSlice, vec::BitVec};

pub struct BloomFilter<T: ?Sized> {
    base: BitVec,
    hashers: [RandomState; 2],
    hashes: u64,
    _marker: PhantomData<T>,
}

impl<T> BloomFilter<T>
where
    T: ?Sized,
{
    #[inline]
    #[must_use]
    pub fn new(capacity: usize, fpr: f64) -> Self {
        assert!(
            (0.0..1.0).contains(&fpr) && fpr > 0.0,
            "false positive rate should be a ratio greater than 0.0"
        );

        // Compute the optimal bitarray size `m` and the optimal number of hash functions `k`
        let m = (-1.0f64 * capacity as f64 * fpr.ln() / (LN_2 * LN_2)).ceil() as usize;
        let k = (-1.0f64 * fpr.ln() / LN_2).ceil() as u64;

        Self {
            base: bitvec![0; max(m, 1)],
            hashers: [RandomState::new(), RandomState::new()],
            hashes: k,
            _marker: PhantomData,
        }
    }

    pub fn from_raw_parts(m: usize, k: u64) -> Self {
        assert!(m > 0 && k > 0, "m and k should be positive");

        Self {
            base: bitvec![0; max(m, 1)],
            hashers: [RandomState::new(), RandomState::new()],
            hashes: k,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn bitslice(&self) -> &BitSlice {
        &self.base
    }
}

impl<T> BloomFilter<T>
where
    T: ?Sized + Hash,
{
    #[inline]
    pub fn contains(&self, value: &T) -> bool {
        let h = (
            self.hashers[0].hash_one(value),
            self.hashers[1].hash_one(value),
        );

        (0..self.hashes).all(|i| {
            let bit =
                usize::try_from(h.0.wrapping_add(i.wrapping_mul(h.1))).unwrap() % self.base.len();
            self.base[bit]
        })
    }

    #[inline]
    pub fn insert(&mut self, value: &T) {
        let h = (
            self.hashers[0].hash_one(value),
            self.hashers[1].hash_one(value),
        );

        (0..self.hashes).for_each(|i| {
            let bit =
                usize::try_from(h.0.wrapping_add(i.wrapping_mul(h.1))).unwrap() % self.base.len();
            self.base.set(bit, true);
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_membership_and_no_false_positives() {
        let mut bloom = BloomFilter::new(100, 0.01);

        assert!(!bloom.contains("1"));
        assert!(!bloom.contains("2"));

        bloom.insert("1");
        assert!(bloom.contains("1"));
    }
}
