use std::marker::PhantomData;

use rand::{Rng, SeedableRng, rngs::StdRng};

use super::symbol::Symbol;

#[derive(Debug)]
pub struct SymbolMapping<'a, T: Symbol> {
    _marker: PhantomData<&'a T>,
    rng: StdRng,
    last_mapped_idx: usize,
}

impl<'a, T: Symbol> SymbolMapping<'a, T> {
    pub fn new(seed: u64) -> Self {
        Self {
            _marker: PhantomData,
            rng: StdRng::seed_from_u64(seed),
            last_mapped_idx: 0,
        }
    }
}

impl<'a, T: Symbol> Iterator for SymbolMapping<'a, T> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        let r: f64 = self.rng.gen_range(0.0..1.0);

        // Calculate the difference based on the random value
        let i = self.last_mapped_idx as f64;
        let u_sqrt_inv = 1.0 / (1.0 - r).sqrt();
        let diff = ((1.5 + i) * ((u_sqrt_inv) - 1.0)).ceil() as usize;

        let index_to_return = self.last_mapped_idx;
        self.last_mapped_idx += diff;
        Some(index_to_return)
    }
}
