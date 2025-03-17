use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    mem,
};

use either::*;
use rand::Rng;

pub trait Decompose {
    type Decomposition;

    fn split(&self) -> Vec<Self::Decomposition>;
    fn join(&mut self, deltas: Vec<Self::Decomposition>);
    fn difference(&self, remote: &Self::Decomposition) -> Self::Decomposition;
}

pub trait Extract {
    type Item: Hash;

    fn extract(&self) -> Self::Item;
}

pub trait Measure {
    fn len(replica: &Self) -> usize
    where
        Self: Decompose;

    fn size_of(replica: &Self) -> usize;
    fn false_matches(&self, other: &Self) -> usize;
}

#[derive(PartialEq, Eq, Debug, Default)]
pub struct Elements<'a, T> {
    elems: Vec<&'a T>,
    idx: usize,
}

impl<'a, T> Iterator for Elements<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.elems.len() {
            return None;
        }

        self.idx += 1;
        Some(self.elems[self.idx - 1])
    }
}

#[derive(Clone, Debug, Default)]
pub struct GSet<T> {
    base: HashSet<T>,
}

impl<T> GSet<T> {
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            base: HashSet::new(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.base.is_empty()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.base.len()
    }
}

impl<T> GSet<T>
where
    T: Eq + Hash,
{
    #[inline]
    pub fn contains(&self, value: &T) -> bool {
        self.base.contains(value)
    }
}

impl<T> GSet<T>
where
    T: Clone + Eq + Hash,
{
    #[inline]
    pub fn elements(&self) -> Elements<'_, T> {
        Elements {
            elems: self.base.iter().collect(),
            idx: 0,
        }
    }

    pub fn insert(&mut self, value: T) -> Self {
        if self.base.insert(value.clone()) {
            Self {
                base: HashSet::from([value]),
            }
        } else {
            Self {
                base: HashSet::new(),
            }
        }
    }
}

impl<T> Decompose for GSet<T>
where
    T: Clone + Eq + Hash,
{
    type Decomposition = GSet<T>;

    fn split(&self) -> Vec<Self::Decomposition> {
        self.base
            .iter()
            .cloned()
            .map(|value| Self {
                base: HashSet::from([value]),
            })
            .collect()
    }

    fn join(&mut self, deltas: Vec<Self::Decomposition>) {
        deltas
            .into_iter()
            .for_each(|delta| self.base.extend(delta.base))
    }

    fn difference(&self, remote: &Self::Decomposition) -> Self::Decomposition {
        Self {
            base: self.base.difference(&remote.base).cloned().collect(),
        }
    }
}

impl<T> Extract for GSet<T>
where
    T: Clone + Eq + Hash,
{
    type Item = T;

    fn extract(&self) -> Self::Item {
        assert_eq!(
            self.len(),
            1,
            "a join-decomposition should have a single item"
        );

        self.base.iter().next().cloned().unwrap()
    }
}

impl Measure for GSet<String> {
    fn len(replica: &Self) -> usize {
        replica.len()
    }

    fn size_of(replica: &Self) -> usize {
        replica.elements().map(String::len).sum()
    }

    fn false_matches(&self, other: &Self) -> usize {
        self.base.symmetric_difference(&other.base).count()
    }
}

impl<T> PartialEq for GSet<T>
where
    T: Eq + Hash,
{
    fn eq(&self, other: &Self) -> bool {
        self.base == other.base
    }
}

impl<T> Eq for GSet<T> where T: Eq + Hash {}

#[cfg(test)]
mod gset {
    use super::*;

    #[test]
    fn test_split_and_join() {
        let mut splittable = GSet::new();

        splittable.insert(1);
        splittable.insert(2);
        splittable.insert(2);
        assert_eq!(splittable.len(), 2);

        let decompositions = splittable.split();
        assert_eq!(decompositions.len(), splittable.len());

        let mut joinable = GSet::new();

        joinable.join(decompositions);
        assert_eq!(joinable.len(), splittable.len());
        assert!(joinable.contains(&1));
        assert!(joinable.contains(&2));

        joinable.insert(3);
    }

    #[test]
    fn test_difference() {
        let local = GSet {
            base: HashSet::from_iter(0..=2),
        };
        let remote = GSet {
            base: HashSet::from_iter(2..=4),
        };

        let diff = local.difference(&remote);
        assert!(diff.contains(&0));
        assert!(diff.contains(&1));
        assert!(!diff.contains(&2));
        assert!(!diff.contains(&3));
        assert!(!diff.contains(&4));
    }

    #[test]
    fn test_difference_synced() {
        let local = GSet {
            base: HashSet::from_iter(0..3),
        };
        let remote = local.clone();

        assert_eq!(local.elements(), remote.elements());

        let diff = local.difference(&remote);
        assert!(diff.is_empty());
    }
}

#[derive(Clone, Debug, Default)]
pub struct AWSet<T> {
    inserted: HashMap<u64, T>,
    removed: HashSet<u64>,
}

impl<T> AWSet<T> {
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            inserted: HashMap::new(),
            removed: HashSet::new(),
        }
    }

    #[inline]
    pub fn elements(&self) -> Elements<'_, T> {
        Elements {
            elems: self
                .inserted
                .iter()
                .filter_map(|(id, v)| (!self.removed.contains(id)).then_some(v))
                .collect(),
            idx: 0,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        !self.inserted.keys().any(|id| !self.removed.contains(id))
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.inserted
            .keys()
            .filter(|id| !self.removed.contains(id))
            .count()
    }

    fn uid(&self) -> u64 {
        let mut rng = rand::thread_rng();
        let mut id = rng.r#gen();

        while self.inserted.contains_key(&id) {
            id = rng.r#gen();
        }

        id
    }
}

impl<T> AWSet<T>
where
    T: Eq + Hash,
{
    #[inline]
    pub fn contains(&self, value: &T) -> bool {
        self.inserted
            .iter()
            .any(|(id, v)| value == v && !self.removed.contains(id))
    }

    pub fn remove(&mut self, value: &T) -> Self {
        let ids = self
            .inserted
            .iter()
            .filter_map(|(id, v)| (value == v && !self.removed.contains(id)).then_some(*id))
            .collect::<HashSet<_>>();

        ids.iter().for_each(|id| {
            self.removed.insert(*id);
        });

        Self {
            inserted: HashMap::new(),
            removed: ids,
        }
    }
}

impl<T> AWSet<T>
where
    T: Clone + Eq + Hash,
{
    pub fn insert(&mut self, value: T) -> Self {
        if !self.contains(&value) {
            let id = self.uid();
            self.inserted.insert(id, value.clone());

            Self {
                inserted: HashMap::from([(id, value)]),
                removed: HashSet::new(),
            }
        } else {
            Self {
                inserted: HashMap::new(),
                removed: HashSet::new(),
            }
        }
    }
}

impl<T> Decompose for AWSet<T>
where
    T: Clone + Eq + Hash,
{
    type Decomposition = AWSet<T>;

    fn split(&self) -> Vec<Self::Decomposition> {
        let inserted = self.inserted.iter().map(|(id, v)| Self {
            inserted: HashMap::from([(*id, v.clone())]),
            removed: HashSet::new(),
        });

        let removed = self.removed.iter().cloned().map(|id| Self {
            inserted: HashMap::new(),
            removed: HashSet::from([id]),
        });

        inserted.chain(removed).collect()
    }

    fn join(&mut self, deltas: Vec<Self::Decomposition>) {
        deltas.into_iter().for_each(|delta| {
            self.inserted.extend(delta.inserted);
            self.removed.extend(delta.removed);
        })
    }

    fn difference(&self, remote: &Self::Decomposition) -> Self::Decomposition {
        Self {
            inserted: self
                .inserted
                .iter()
                .filter(|(id, _)| !remote.inserted.contains_key(id))
                .map(|(id, v)| (*id, v.clone()))
                .collect(),
            removed: self.removed.difference(&remote.removed).cloned().collect(),
        }
    }
}

impl<T> Extract for AWSet<T>
where
    T: Clone + Eq + Hash,
{
    type Item = Either<(u64, T), u64>;

    fn extract(&self) -> Self::Item {
        if self.removed.is_empty() {
            assert_eq!(
                self.inserted.len(),
                1,
                "a join-decomposition should have a single item"
            );

            Left(
                self.inserted
                    .iter()
                    .map(|(id, v)| (*id, v.clone()))
                    .next()
                    .unwrap(),
            )
        } else {
            assert_eq!(
                self.removed.len(),
                1,
                "a join-decomposition should have a single item"
            );

            Right(self.removed.iter().cloned().next().unwrap())
        }
    }
}

impl Measure for AWSet<String> {
    fn len(replica: &Self) -> usize {
        replica.inserted.len() + replica.removed.len()
    }

    fn size_of(replica: &Self) -> usize {
        replica.inserted.len() * mem::size_of::<u64>()
            + replica.inserted.values().map(String::len).sum::<usize>()
            + replica.removed.len() * mem::size_of::<u64>()
    }

    fn false_matches(&self, other: &Self) -> usize {
        self.elements().filter(|v| !other.contains(v)).count()
            + other.elements().filter(|v| !self.contains(v)).count()
    }
}

impl<T> PartialEq for AWSet<T>
where
    T: Eq + Hash,
{
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }

        self.inserted
            .iter()
            .filter_map(|(id, v)| (!self.removed.contains(id)).then_some(v))
            .all(|id| other.contains(id))
    }
}

impl<T> Eq for AWSet<T> where T: Eq + Hash {}

#[cfg(test)]
mod awset {
    use super::*;

    #[test]
    fn test_insert_and_remove() {
        let mut awset = AWSet::new();
        assert_eq!(awset.len(), 0);
        assert!(awset.is_empty());

        awset.insert(1);
        awset.insert(2);
        awset.insert(3);
        assert_eq!(awset.len(), 3);
        assert!(!awset.is_empty());

        awset.remove(&2);
        awset.remove(&2);
        awset.remove(&4);
        assert_eq!(awset.len(), 2);

        awset.insert(2);
        awset.insert(4);
        assert_eq!(awset.len(), 4);
    }

    #[test]
    fn test_elements() {
        let mut awset = AWSet::new();
        awset.insert(1);
        awset.insert(2);
        awset.insert(3);
        awset.insert(3);

        assert_eq!(awset.elements().count(), 3);
        assert!(awset.elements().all(|v| vec![1, 2, 3].contains(v)));

        awset.remove(&1);
        awset.insert(3);
        awset.remove(&3);

        assert_eq!(awset.elements().next(), Some(&2));

        awset.remove(&2);
        assert_eq!(awset.elements().next(), None);
    }

    #[test]
    fn test_split_and_join() {
        let mut splittable = AWSet::new();

        splittable.insert(1);
        splittable.insert(2);
        splittable.insert(3);
        splittable.remove(&2);
        splittable.remove(&4);

        assert!(splittable.contains(&1));
        assert!(splittable.contains(&3));

        let decompositions = splittable.split();
        assert_eq!(decompositions.len(), 4);

        let mut joinable = AWSet::new();
        joinable.join(decompositions);

        assert_eq!(splittable, joinable);
    }

    #[test]
    fn test_difference() {
        let local = AWSet {
            inserted: HashMap::from([(1, 1), (2, 3), (3, 2), (4, 4), (5, 10)]),
            removed: HashSet::from([1, 3]),
        };

        let remote = AWSet {
            inserted: HashMap::from([(1, 1), (2, 3), (3, 2)]),
            removed: HashSet::from([1, 2]),
        };

        let diff = local.difference(&remote);
        assert_eq!(diff.inserted, HashMap::from([(4, 4), (5, 10)]));
        assert_eq!(diff.removed, HashSet::from([3]));
    }

    #[test]
    fn test_difference_synced() {
        let local = AWSet {
            inserted: HashMap::from([(1, 1), (2, 3), (3, 2), (4, 4), (5, 10)]),
            removed: HashSet::from([1, 3]),
        };

        let remote = AWSet {
            inserted: HashMap::from([(1, 1), (2, 3), (3, 2), (4, 4), (5, 10)]),
            removed: HashSet::from([1, 3]),
        };

        assert_eq!(local, remote);

        let diff = local.difference(&remote);
        assert!(diff.inserted.is_empty());
        assert!(diff.removed.is_empty());
    }

    #[test]
    fn test_false_matches() {
        let local = AWSet {
            inserted: HashMap::from([
                (1, "1".to_string()),
                (4, "4".to_string()),
                (5, "10".to_string()),
            ]),
            removed: HashSet::from([1, 4]),
        };

        let remote = AWSet {
            inserted: HashMap::from([
                (1, "1".to_string()),
                (2, "3".to_string()),
                (3, "2".to_string()),
            ]),
            removed: HashSet::from([1, 2]),
        };

        let local_elems = local.elements().collect::<HashSet<_>>();
        let remote_elems = remote.elements().collect::<HashSet<_>>();
        assert_eq!(
            local.false_matches(&remote),
            local_elems.symmetric_difference(&remote_elems).count()
        )
    }
}
