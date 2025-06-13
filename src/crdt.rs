use std::{
    cmp::max, collections::{HashMap, HashSet}, hash::Hash, mem
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


#[derive(Clone, Debug, Default)]
pub struct PNCounter<I> {
    inner: HashMap<I, (u64, u64)>,
}

impl<I> PNCounter<I>{
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: HashMap::default(),
        }
    }
}

impl<I> PNCounter<I>
where
    I: Eq + Hash,
{
    pub fn count(&self) -> i64 {
        self.inner.values().map(|(incr, decr)| *incr as i64 - *decr as i64).sum()
    }
}


impl<I> PNCounter<I>
where
    I: Clone + Eq + Hash,
{
    pub fn increment(&mut self, id: &I) -> Self {
        self.add(id, 1)
    }

    pub fn add(&mut self, id: &I, value: u64) -> Self{
        match self.inner.get_mut(id) {
            Some((incr, decr)) => *incr += value,
            None => {
                self.inner.insert(id.clone(), (value, 0));
            }
        }

        let (key_ref, value_ref) = self
        .inner
        .get_key_value(id)
        .expect("key not found in counter");
    
        let entry = (key_ref.clone(), value_ref.clone());

        Self {
            inner: HashMap::from([entry]),
        }
    }

    pub fn decrement(&mut self, id: &I) -> Self {
        self.sub(id, 1)
    }

    pub fn sub(&mut self, id: &I, value: u64) -> Self{
        match self.inner.get_mut(id) {
            Some((incr, decr)) => *decr += value,
            None => {
                self.inner.insert(id.clone(), (0, value));
            }
        }

        let (key_ref, value_ref) = self
        .inner
        .get_key_value(id)
        .expect("key not found in counter");
    
        let entry = (key_ref.clone(), value_ref.clone());

        Self {
            inner: HashMap::from([entry]),
        }
    }
}


impl<I> Decompose for PNCounter<I>
where
    I: Clone + Eq + Hash,
{
    type Decomposition = PNCounter<I>;

    fn split(&self) -> Vec<Self::Decomposition> {
        self.inner.iter().map(|(id, v)| Self {
            inner: HashMap::from([(id.clone(), v.clone())]),
        }).collect()
    }

    fn join(&mut self, deltas: Vec<Self::Decomposition>) {
        for delta in deltas {
            for (id, val) in &delta.inner {
                let (delta_incr,delta_decr) = val;
                self.inner
                    .entry(id.clone())
                    .and_modify(|(incr,decr)| {
                        *incr = max(*incr, *delta_incr);
                        *decr = max(*decr, *delta_decr);
                    })
                    .or_insert_with(|| val.clone());
            }
        }
    }

    fn difference(&self, remote: &Self::Decomposition) -> Self::Decomposition {
        let mut result = HashMap::new();
    
        for (id, val) in &self.inner {
            let (incr_local, decr_local) = val;
            match remote.inner.get(id) {
                Some((incr_remote, decr_remote)) 
                    if ! (incr_local <= incr_remote && decr_local <= decr_remote) => { 
                        //if local is not less than or equal to remote
                    result.insert(id.clone(), val.clone());
                }
                None => {
                    result.insert(id.clone(), val.clone());
                }
                _ => {}
            }
        }
    
        Self::Decomposition { inner: result }
    }
}
    

impl<I> Extract for PNCounter<I>
where
    I: Clone + Eq + Hash,
{
    type Item = (I, (u64,u64));

    fn extract(&self) -> Self::Item {
        assert_eq!(
            self.inner.len(),
            1,
            "a join-decomposition should have a single item"
        );


        let (id, val) = self.inner.iter().next().unwrap();

        (id.clone(), val.clone())
    }
}



impl Measure for PNCounter<String> {
    fn len(replica: &Self) -> usize {
        replica.inner.len()
    }

    fn size_of(replica: &Self) -> usize {
        replica.inner.len() * 2 * mem::size_of::<u64>()
            + replica.inner.keys().map(String::len).sum::<usize>()
    }

    fn false_matches(&self, other: &Self) -> usize {
        let only_in_self = self.inner.iter()
            .filter(|(id, val_self)| {
                match other.inner.get(*id) {
                    Some(val_other) => *val_self != val_other,
                    None => true,
                }
            })
            .count();
    
        let only_in_other = other.inner.keys()
            .filter(|id| !self.inner.contains_key(*id))
            .count();
    
        only_in_self + only_in_other
    }
}


impl<I> PartialEq for PNCounter<I>
where
    I: Eq + Hash,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<I> Eq for PNCounter<I> where I: Eq + Hash {}

#[cfg(test)]
mod pncounter {
    use super::*;

    #[test]
    fn test_increment_and_decrement() {
        let mut counter = PNCounter::new();
        assert_eq!(counter.count(), 0);

        counter.increment(&"a".to_string());
        assert_eq!(counter.count(), 1);

        counter.add(&"b".to_string(), 5);
        assert_eq!(counter.count(), 6);

        counter.decrement(&"a".to_string());
        assert_eq!(counter.count(), 5);

        counter.sub(&"b".to_string(), 2);
        assert_eq!(counter.count(), 3);

        counter.increment(&"c".to_string());
        assert_eq!(counter.count(), 4);

        counter.sub(&"d".to_string(), 10); // Decrement a non-existent key
        assert_eq!(counter.count(), -6);
    }

    #[test]
    fn test_count() {
        let mut counter = PNCounter::new();
        counter.increment(&"item1".to_string()); // +1
        counter.increment(&"item1".to_string()); // +1
        counter.decrement(&"item1".to_string()); // -1
        counter.increment(&"item2".to_string()); // +1
        counter.sub(&"item3".to_string(), 5); // -5

        assert_eq!(counter.count(), -3); // (1+1-1) + 1 + (-5) = 1 + 1 - 5 = -3
    }

    #[test]
    fn test_split_and_join() {
        let mut splittable = PNCounter::new();
        splittable.increment(&"a".to_string());
        splittable.add(&"b".to_string(), 3);
        splittable.decrement(&"a".to_string());
        splittable.sub(&"c".to_string(), 2);

        let initial_count = splittable.count();
        let decompositions = splittable.split();
        assert_eq!(decompositions.len(), splittable.inner.len());

        let mut joinable = PNCounter::new();
        joinable.join(decompositions);

        assert_eq!(joinable.count(), initial_count);
        assert_eq!(joinable, splittable);
    }

    #[test]
    fn test_difference() {
        let local = PNCounter {
            inner: HashMap::from([
                ("a".to_string(), (5, 2)),
                ("b".to_string(), (3, 1)),
                ("c".to_string(), (0, 4)),
            ]),
        };

        let remote = PNCounter {
            inner: HashMap::from([
                ("a".to_string(), (4, 2)),
                ("b".to_string(), (3, 2)),
                ("d".to_string(), (1, 0)),
            ]),
        };

        let diff = local.difference(&remote);

        let expected_inner = HashMap::from([
            ("a".to_string(), (5, 2)),
            ("c".to_string(), (0, 4)),
        ]);
        assert_eq!(diff.inner, expected_inner);
    }

    #[test]
    fn test_difference_synced() {
        let local = PNCounter {
            inner: HashMap::from([
                ("a".to_string(), (5, 2)),
                ("b".to_string(), (3, 1)),
            ]),
        };
        let remote = local.clone();

        let diff = local.difference(&remote);
        assert!(diff.inner.is_empty());
    }

    #[test]
    fn test_false_matches() {
        let local = PNCounter {
            inner: HashMap::from([
                ("a".to_string(), (5, 2)),
                ("b".to_string(), (3, 1)),
                ("c".to_string(), (0, 4)),
            ]),
        };

        let remote = PNCounter {
            inner: HashMap::from([
                ("a".to_string(), (5, 2)), // Matches exactly
                ("b".to_string(), (3, 2)), // Mismatch in decrement
                ("d".to_string(), (1, 0)), // Only in remote
            ]),
        };

        // "a" matches
        // "b" mismatches (value differs) -> 1 false match
        // "c" only in local -> 1 false match
        // "d" only in remote -> 1 false match

        assert_eq!(local.false_matches(&remote), 3);
    }

    #[test]
    fn test_join() {
        let mut counter1 = PNCounter::new();
        counter1.increment(&"a".to_string()); // a: (1, 0)
        counter1.add(&"b".to_string(), 5);    // b: (5, 0)

        let mut counter2 = PNCounter::new();
        counter2.increment(&"a".to_string()); // a: (1, 0)
        counter2.add(&"a".to_string(), 2);    // a: (3, 0)
        counter2.decrement(&"b".to_string()); // b: (0, 1)
        counter2.sub(&"c".to_string(), 4);    // c: (0, 4)

        // Simulate deltas that would come from splitting other replicas
        let delta1 = PNCounter {
            inner: HashMap::from([
                ("a".to_string(), (1, 0)),
                ("b".to_string(), (5, 0)),
            ]),
        };

        let delta2 = PNCounter {
            inner: HashMap::from([
                ("a".to_string(), (3, 0)), // Higher increment for 'a'
                ("b".to_string(), (0, 1)), // Higher decrement for 'b'
                ("c".to_string(), (0, 4)),
            ]),
        };

        let mut combined_counter = PNCounter::new();
        combined_counter.join(vec![delta1, delta2]);

        // Expected state after join:
        // 'a': max(1, 3) = 3 for increment, max(0, 0) = 0 for decrement => (3, 0)
        // 'b': max(5, 0) = 5 for increment, max(0, 1) = 1 for decrement => (5, 1)
        // 'c': max(0, 0) = 0 for increment, max(0, 4) = 4 for decrement => (0, 4)
        let expected_inner = HashMap::from([
            ("a".to_string(), (3, 0)),
            ("b".to_string(), (5, 1)),
            ("c".to_string(), (0, 4)),
        ]);

        assert_eq!(combined_counter.inner, expected_inner);
        assert_eq!(combined_counter.count(), 3 + 5 - 1 - 4); // 3 for 'a', 4 for 'b', -4 for 'c' = 3
    }
}