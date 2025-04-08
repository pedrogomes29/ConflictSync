use std::{collections::HashMap, fmt::Display, hash::RandomState, marker::PhantomData, mem};

use crate::{
    crdt::{Decompose, Extract, Measure},
    riblt::RatelessIBLT,
    tracker::{DefaultEvent, DefaultTracker, Telemetry},
};

use super::{Algorithm, Dispatcher};

#[derive(Clone, Copy, Debug)]
pub struct RibltBuckets<T> {
    lf: f64,
    _marker: PhantomData<T>,
}

impl<T> RibltBuckets<T> {
    #[inline]
    #[must_use]
    pub fn new(lf: f64) -> Self {
        assert!(lf > 0.0, "load factor should be greater than 0.0");

        Self {
            lf,
            _marker: PhantomData,
        }
    }
}

impl<T> Default for RibltBuckets<T> {
    fn default() -> Self {
        Self {
            lf: 1.0,
            _marker: PhantomData,
        }
    }
}

impl<T> Display for RibltBuckets<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Bucketing+Rateless[lf={}]", self.lf)
    }
}

impl<T> Dispatcher<T> for RibltBuckets<T> where T: Clone + Decompose<Decomposition = T> + Extract {}

impl<T> Algorithm<T> for RibltBuckets<T>
where
    T: Clone + Decompose<Decomposition = T> + Default + Extract + Measure,
{
    type Tracker = DefaultTracker;

    fn sync(&self, local: &mut T, remote: &mut T, tracker: &mut Self::Tracker) {
        const CODED_SYMBOL_SIZE: usize =
            mem::size_of::<u64>() + mem::size_of::<u64>() + mem::size_of::<i64>();
        assert!(
            tracker.is_ready(),
            "tracker should be ready, i.e., no captured events and not finished"
        );

        let hasher = RandomState::new();
        let buckets = (self.lf * <T as Measure>::len(local) as f64) as usize;

        // 1. Assign each join-decomposition to a bucket based on the modulo of its hash
        //    Create a rateless IBLT from the hash of the buckets
        // NOTE: This policy must be deterministic across both peers.
        let local_buckets = self.dispatch(local, buckets, &hasher);
        let local_hashes = RibltBuckets::<T>::hashes_to_bucket_index(&local_buckets, &hasher);
        let mut local_iblt = RatelessIBLT::riblt_from(local_hashes.keys().cloned());

        // 2. Repeat the procedure from 1., but now on the remote replica.
        let remote_buckets = self.dispatch(remote, buckets, &hasher);
        let remote_hashes = RibltBuckets::<T>::hashes_to_bucket_index(&remote_buckets, &hasher);
        let mut remote_iblt = RatelessIBLT::riblt_from(remote_hashes.keys().cloned());

        // 3. Send Coded symbols until the remote replica has enough to decode all the differences
        remote_iblt.find_all_differences(&mut local_iblt);
        let sketch_size = local_iblt.sketch.len();
        assert_eq!(sketch_size, remote_iblt.sketch.len());

        tracker.register(DefaultEvent::LocalToRemote {
            state: 0,
            metadata: sketch_size * CODED_SYMBOL_SIZE,
            upload: tracker.upload(),
        });

        // 4. Compute the buckets whose hash does not match on the remote replica and send those
        //    buckets back to the local replica.
        let remote_only_hashes = remote_iblt.get_local_only_symbols();
        let non_matching = remote_only_hashes
            .iter()
            .map(|hash| {
                let bucket_idx = remote_hashes[hash];
                let bucket = &remote_buckets[bucket_idx];
                let mut state = T::default();
                state.join(bucket.values().cloned().collect());
                (bucket_idx, state)
            })
            .collect::<HashMap<_, _>>();

        tracker.register(DefaultEvent::RemoteToLocal {
            state: non_matching.values().map(<T as Measure>::size_of).sum(),
            metadata: non_matching.keys().count() * mem::size_of::<usize>(),
            download: tracker.download(),
        });

        // 5. Compute the differences between buckets against both the local and remote
        //    decompositions. Then send the difference unknown by remote replica.
        let remote_buckets = non_matching;
        let local_buckets = local_buckets
            .into_iter()
            .enumerate()
            .filter_map(|(i, bucket)| {
                remote_buckets.contains_key(&i).then(|| {
                    let mut state = T::default();
                    state.join(bucket.into_values().collect());

                    (i, state)
                })
            })
            .collect::<HashMap<_, _>>();

        let local_unknown = local_buckets
            .iter()
            .map(|(i, local)| remote_buckets.get(i).unwrap().difference(local));
        let remote_unknown = remote_buckets
            .iter()
            .map(|(i, remote)| local_buckets.get(i).unwrap().difference(remote))
            .collect::<Vec<_>>();

        tracker.register(DefaultEvent::LocalToRemote {
            state: remote_unknown.iter().map(<T as Measure>::size_of).sum(),
            metadata: 0,
            upload: tracker.upload(),
        });

        // 6. Join the appropriate join-decompositions to each replica.
        local.join(local_unknown.collect());
        remote.join(remote_unknown);

        // 7. Sanity check.
        tracker.finish(<T as Measure>::false_matches(local, remote));
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::{crdt::GSet, tracker::Bandwidth};

    #[test]
    fn test_sync() {
        let mut local = {
            let mut gset = GSet::new();
            let items = "Stuck In A Moment You Can't Get Out Of"
                .split_whitespace()
                .collect::<Vec<_>>();

            for item in items {
                gset.insert(item.to_string());
            }

            gset
        };

        let mut remote = {
            let mut gset = GSet::new();
            let items = "I Still Haven't Found What I'm Looking For"
                .split_whitespace()
                .collect::<Vec<_>>();

            for item in items {
                gset.insert(item.to_string());
            }

            gset
        };

        let (download, upload) = (Bandwidth::Kbps(0.5), Bandwidth::Kbps(0.5));
        let mut tracker = DefaultTracker::new(download, upload);
        let buckets = RibltBuckets::new(1.25);

        buckets.sync(&mut local, &mut remote, &mut tracker);

        assert_eq!(tracker.false_matches(), 0);
    }
}
