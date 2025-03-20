mod mapping;
mod symbol;

use std::{
    cmp::{Ordering, Reverse},
    collections::{BinaryHeap, HashSet},
    hash::{DefaultHasher, Hasher},
};

use mapping::SymbolMapping;
use symbol::{CodedSymbol, Direction, HashedSymbol};
pub use symbol::Symbol;


#[derive(Debug)]
struct HashedSymbolMapping<'a, T: Symbol> {
    hashed_symbol: HashedSymbol<T>,
    mapping: SymbolMapping<'a, T>,
}

struct SourceSymbolIdxToLastMappingIdx {
    source_symbol_idx: usize,
    last_mapping_idx: usize,
}

impl PartialEq for SourceSymbolIdxToLastMappingIdx {
    fn eq(&self, other: &Self) -> bool {
        self.last_mapping_idx == other.last_mapping_idx
    }
}

impl Eq for SourceSymbolIdxToLastMappingIdx {}

impl PartialOrd for SourceSymbolIdxToLastMappingIdx {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SourceSymbolIdxToLastMappingIdx {
    fn cmp(&self, other: &Self) -> Ordering {
        self.last_mapping_idx.cmp(&other.last_mapping_idx)
    }
}

#[derive(Debug)]
enum DecodedSymbolIdx {
    LocalIdx(usize),
    RemoteIdx(usize),
}
struct DecodedSymbolIdxToLastMappingIdx {
    decoded_symbol_idx: DecodedSymbolIdx,
    last_mapping_idx: usize,
}

impl PartialEq for DecodedSymbolIdxToLastMappingIdx {
    fn eq(&self, other: &Self) -> bool {
        self.last_mapping_idx == other.last_mapping_idx
    }
}

impl Eq for DecodedSymbolIdxToLastMappingIdx {}

impl PartialOrd for DecodedSymbolIdxToLastMappingIdx {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DecodedSymbolIdxToLastMappingIdx {
    fn cmp(&self, other: &Self) -> Ordering {
        self.last_mapping_idx.cmp(&other.last_mapping_idx)
    }
}
pub struct Sketch<T: Symbol> {
    coded_symbols: Vec<CodedSymbol<T>>,
}

impl<T: Symbol> Sketch<T> {
    fn new() -> Self {
        Self {
            coded_symbols: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        return self.coded_symbols.len();
    }
}

pub struct RatelessIBLT<'a, T: Symbol> {
    pub sketch: Sketch<T>,
    subtracted_index: Option<usize>,
    source_symbols: Vec<HashedSymbolMapping<'a, T>>,
    next_mapping_idx: BinaryHeap<Reverse<SourceSymbolIdxToLastMappingIdx>>,
    next_peeling_idx: BinaryHeap<Reverse<DecodedSymbolIdxToLastMappingIdx>>,
    local_only: Vec<HashedSymbolMapping<'a, T>>,
    remote_only: Vec<HashedSymbolMapping<'a, T>>,
    decoded: HashSet<T>,
}

impl<'a, T: Symbol> RatelessIBLT<'a, T> {
    pub fn new() -> Self {
        RatelessIBLT {
            sketch: Sketch::<T>::new(),
            subtracted_index: None,
            source_symbols: Vec::new(),
            next_mapping_idx: BinaryHeap::new(),
            next_peeling_idx: BinaryHeap::new(),
            local_only: Vec::new(),
            remote_only: Vec::new(),
            decoded: HashSet::new(),
        }
    }

    pub fn riblt_from<I>(symbols: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        symbols.into_iter().fold(RatelessIBLT::new(), |mut riblt, symbol| {
            riblt.add_symbol(symbol);
            riblt
        })
    }
    
    pub fn add_symbol(&mut self, t: T) {
        let hashed_symbol = HashedSymbol::new(t);
        let mut mapping = SymbolMapping::new(hashed_symbol.hash);
        let first_mapping_idx = mapping.next().expect("Mapping should be infinite");

        self.source_symbols.push(HashedSymbolMapping {
            mapping,
            hashed_symbol,
        });

        self.next_mapping_idx
            .push(Reverse(SourceSymbolIdxToLastMappingIdx {
                source_symbol_idx: self.source_symbols.len() - 1,
                last_mapping_idx: first_mapping_idx,
            }));
    }

    fn peel(&mut self) {
        let sketch_len = self.sketch.coded_symbols.len();
        while let Some(Reverse(DecodedSymbolIdxToLastMappingIdx {
            decoded_symbol_idx,
            last_mapping_idx,
        })) = self.next_peeling_idx.pop()
        {
            if last_mapping_idx >= sketch_len {
                //TODO: use peek() in loop condition (O(1)) and avoid pop/push (O(logn))
                self.next_peeling_idx
                    .push(Reverse(DecodedSymbolIdxToLastMappingIdx {
                        decoded_symbol_idx,
                        last_mapping_idx,
                    }));
                break;
            }

            let (decoded_symbol, direction) = match decoded_symbol_idx {
                DecodedSymbolIdx::LocalIdx(idx) => {
                    (self.local_only.get_mut(idx).unwrap(), Direction::Remove)
                }
                DecodedSymbolIdx::RemoteIdx(idx) => {
                    (self.remote_only.get_mut(idx).unwrap(), Direction::Add)
                }
            };

            let next_coded_symbol = self.sketch.coded_symbols.get_mut(last_mapping_idx).unwrap();
            next_coded_symbol.apply(decoded_symbol.hashed_symbol.clone(), direction);

            self.next_peeling_idx
                .push(Reverse(DecodedSymbolIdxToLastMappingIdx {
                    decoded_symbol_idx,
                    last_mapping_idx: decoded_symbol
                        .mapping
                        .next()
                        .expect("Mapping should be infinite"),
                }));

            self.try_mark_pure_cell(last_mapping_idx);
        }
    }

    fn try_mark_pure_cell(&mut self, idx: usize) {
        let c1 = &mut self.sketch.coded_symbols[idx];
        let mut hasher = DefaultHasher::new();
        c1.hashed_symbol.symbol.hash(&mut hasher);
        let symbol_hash = hasher.finish();
        if symbol_hash == c1.hashed_symbol.hash && !self.decoded.contains(&c1.hashed_symbol.symbol)
        {
            match c1.count {
                1 => {
                    let hashed_symbol = c1.hashed_symbol.clone();
                    self.decoded.insert(hashed_symbol.symbol.clone());

                    let mut mapping = SymbolMapping::new(hashed_symbol.hash);
                    let first_mapping_idx = mapping.next().expect("Mapping should be infinite");

                    self.local_only.push(HashedSymbolMapping {
                        mapping,
                        hashed_symbol,
                    });

                    let new_symbol_idx = self.local_only.len() - 1;

                    self.next_peeling_idx
                        .push(Reverse(DecodedSymbolIdxToLastMappingIdx {
                            decoded_symbol_idx: DecodedSymbolIdx::LocalIdx(new_symbol_idx),
                            last_mapping_idx: first_mapping_idx,
                        }))
                }
                -1 => {
                    let hashed_symbol = c1.hashed_symbol.clone();
                    self.decoded.insert(hashed_symbol.symbol.clone());
                    let mut mapping = SymbolMapping::new(hashed_symbol.hash);
                    let first_mapping_idx = mapping.next().expect("Mapping should be infinite");

                    self.remote_only.push(HashedSymbolMapping {
                        mapping,
                        hashed_symbol,
                    });

                    let new_symbol_idx = self.remote_only.len() - 1;

                    self.next_peeling_idx
                        .push(Reverse(DecodedSymbolIdxToLastMappingIdx {
                            decoded_symbol_idx: DecodedSymbolIdx::RemoteIdx(new_symbol_idx),
                            last_mapping_idx: first_mapping_idx,
                        }));
                }
                _ => (),
            }
        }
    }

    pub fn subtract(&mut self, s2: &Sketch<T>) {
        assert_eq!(
            self.sketch.coded_symbols.len(),
            s2.coded_symbols.len(),
            "Subtracting sketches of different sizes"
        );

        let start = match self.subtracted_index {
            Some(subtracted_index) => subtracted_index + 1,
            None => 0,
        };

        if start >= self.sketch.coded_symbols.len() {
            return;
        }

        for i in start..self.sketch.coded_symbols.len() {
            let c1 = &mut self.sketch.coded_symbols[i];
            let c2 = &s2.coded_symbols[i];

            c1.hashed_symbol.symbol ^= c2.hashed_symbol.symbol.clone();
            c1.count -= c2.count;
            c1.hashed_symbol.hash ^= c2.hashed_symbol.hash;

            //adds pure cells to a heap from which they will be used for peeling
            self.try_mark_pure_cell(i);
            self.peel();
        }

        self.subtracted_index = Some(self.sketch.coded_symbols.len() - 1);
    }

    pub fn extend_sketch(&mut self, extra_size: usize) {
        let current_len = self.sketch.coded_symbols.len();
        let extended_len = current_len + extra_size;

        for _ in current_len..current_len + extra_size {
            self.sketch.coded_symbols.push(CodedSymbol::<T>::new());
        }

        while let Some(Reverse(SourceSymbolIdxToLastMappingIdx {
            source_symbol_idx,
            last_mapping_idx,
        })) = self.next_mapping_idx.pop()
        {
            if last_mapping_idx >= extended_len {
                //TODO: use peek() in loop condition (O(1)) and avoid pop/push (O(logn))
                self.next_mapping_idx
                    .push(Reverse(SourceSymbolIdxToLastMappingIdx {
                        source_symbol_idx,
                        last_mapping_idx,
                    }));
                break;
            }

            let coded_symbol = self.sketch.coded_symbols.get_mut(last_mapping_idx).unwrap();
            let original_symbol = self.source_symbols.get_mut(source_symbol_idx).unwrap();
            coded_symbol.apply(original_symbol.hashed_symbol.clone(), Direction::Add);

            self.next_mapping_idx
                .push(Reverse(SourceSymbolIdxToLastMappingIdx {
                    source_symbol_idx,
                    last_mapping_idx: original_symbol
                        .mapping
                        .next()
                        .expect("Mapping should be infinite"),
                }))
        }
    }

    pub fn is_decoded(&self) -> bool {
        //all source symbols map to the first coded symbol
        //if it is decoded, all symbols have been decoded
        let first_symbol = &self.sketch.coded_symbols[0];
        first_symbol.hashed_symbol.hash == 0
            && first_symbol.hashed_symbol.symbol == T::default()
            && first_symbol.count == 0
    }

    pub fn get_local_only_symbols(&self) -> Vec<T> {
        self.local_only
            .iter()
            .map(|hashed_symbol_mapping| hashed_symbol_mapping.hashed_symbol.symbol.clone())
            .collect()
    }

    pub fn get_remote_only_symbols(&self) -> Vec<T> {
        self.remote_only
            .iter()
            .map(|hashed_symbol_mapping| hashed_symbol_mapping.hashed_symbol.symbol.clone())
            .collect()
    }

    pub fn find_all_differences(&mut self, iblt2: &mut RatelessIBLT<T>) -> usize {
        self.extend_sketch(1);
        iblt2.extend_sketch(1);
        loop {
            self.subtract(&iblt2.sketch);
            if self.is_decoded() {
                return self.sketch.coded_symbols.len();
            }
            self.extend_sketch(1);
            iblt2.extend_sketch(1);
        }
    }
}

impl Symbol for u64 {}

#[cfg(test)]
mod tests {
    use super::*;

    impl Symbol for i32 {}

    #[test]
    fn test_rateless_iblt_subtract_and_decode() {
        // Create two rateless IBLTs with different symbol sets
        let mut iblt1 = RatelessIBLT::<i32>::new();
        let mut iblt2 = RatelessIBLT::<i32>::new();

        for i in 1..=100 {
            iblt1.add_symbol(i);
        }

        for i in 2..=101 {
            iblt2.add_symbol(i);
        }

        let sketch_size = iblt1.find_all_differences(&mut iblt2);
        println!("Sketch size required: {sketch_size}");

        let local_only_symbols: Vec<i32> = iblt1.get_local_only_symbols();
        assert!(
            local_only_symbols.contains(&1),
            "Expected 1 in local_only, but found {:?}",
            local_only_symbols
        );

        let remote_only_symbols: Vec<i32> = iblt1.get_remote_only_symbols();
        assert!(
            remote_only_symbols.contains(&101),
            "Expected 101 in remote_only, but found {:?}",
            remote_only_symbols
        );
    }

    #[test]
    fn test_rateless_iblt_complex_subtract_and_decode() {
        let mut iblt1 = RatelessIBLT::<i32>::new();
        let mut iblt2 = RatelessIBLT::<i32>::new();

        // iblt1 contains {1, 2, 3, ..., 50, 101, 102, 103}
        for i in 1..=50 {
            iblt1.add_symbol(i);
        }
        iblt1.add_symbol(101);
        iblt1.add_symbol(102);
        iblt1.add_symbol(103);

        // iblt2 contains {25, 26, ..., 75, 200, 201}
        for i in 25..=75 {
            iblt2.add_symbol(i);
        }
        iblt2.add_symbol(200);
        iblt2.add_symbol(201);

        let sketch_size = iblt1.find_all_differences(&mut iblt2);
        println!("Sketch size required: {sketch_size}");

        let local_only_symbols: Vec<i32> = iblt1.get_local_only_symbols();
        let remote_only_symbols: Vec<i32> = iblt1.get_remote_only_symbols();

        // Expected local-only: {1, 2, ..., 24, 101, 102, 103}
        let expected_local: Vec<i32> = (1..25).chain([101, 102, 103]).collect();
        for &symbol in &expected_local {
            assert!(
                local_only_symbols.contains(&symbol),
                "Missing {symbol} in local-only symbols"
            );
        }

        // Expected remote-only: {51, ..., 75, 200, 201}
        let expected_remote: Vec<i32> = (51..=75).chain([200, 201]).collect();
        for &symbol in &expected_remote {
            assert!(
                remote_only_symbols.contains(&symbol),
                "Missing {symbol} in remote-only symbols"
            );
        }

        println!("Local-only symbols found: {:?}", local_only_symbols);
        println!("Remote-only symbols found: {:?}", remote_only_symbols);
    }
}
