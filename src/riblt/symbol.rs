use std::collections::hash_map::DefaultHasher;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::ops::{BitXor, BitXorAssign};

#[derive(Clone, Copy)]
pub enum Direction {
    Add,
    Remove,
}

impl Direction {
    fn as_int(self) -> i64 {
        match self {
            Direction::Add => 1,
            Direction::Remove => -1,
        }
    }
}

pub trait Symbol:
    Hash + BitXor + BitXorAssign + PartialEq + Clone + Default + Debug + std::cmp::Eq
{
}

#[derive(Clone, Debug)]
pub struct HashedSymbol<T: Symbol> {
    pub symbol: T,
    pub hash: u64,
}

impl<T: Symbol> HashedSymbol<T> {
    pub fn new(symbol: T) -> Self {
        let mut hasher = DefaultHasher::new();
        symbol.hash(&mut hasher);
        let hash = hasher.finish();
        Self { symbol, hash }
    }
}

#[derive(Clone, Debug)]
pub struct CodedSymbol<T: Symbol> {
    pub hashed_symbol: HashedSymbol<T>,
    pub count: i64,
}

impl<T: Symbol> CodedSymbol<T> {
    pub fn new() -> Self {
        Self {
            hashed_symbol: HashedSymbol {
                symbol: T::default(),
                hash: 0,
            },
            count: 0,
        }
    }

    pub fn apply(&mut self, s: HashedSymbol<T>, direction: Direction) {
        self.hashed_symbol.symbol ^= s.symbol;
        self.hashed_symbol.hash ^= s.hash;
        self.count += direction.as_int();
    }
}
