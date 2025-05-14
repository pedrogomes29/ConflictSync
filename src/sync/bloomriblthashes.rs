use std::{
    collections::HashMap,
    fmt::Display,
    hash::{BuildHasher, RandomState},
    marker::PhantomData,
    mem,
};

use crate::{
    crdt::{Decompose, Extract, Measure},
    riblt::RatelessIBLT,
    tracker::{DefaultEvent, DefaultTracker, Telemetry},
};

use super::{Algorithm, BuildFilter, Dispatcher};

#[derive(Clone, Copy, Debug)]
pub struct BloomRibltHashes<T> {
    fpr: f64,
    _marker: PhantomData<T>,
}

impl<T> BloomRibltHashes<T> {
    #[inline]
    #[must_use]
    pub fn new(fpr: f64) -> Self {
        assert!(
            fpr > 0.0 && (0.0..1.0).contains(&fpr),
            "fpr should be a ratio in the interval (0.0, 1.0)"
        );

        Self {
            fpr,
            _marker: PhantomData,
        }
    }
}

impl<T> Default for BloomRibltHashes<T> {
    fn default() -> Self {
        Self {
            fpr: 0.01,
            _marker: PhantomData,
        }
    }
}

impl<T> Display for BloomRibltHashes<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Bloom+Rateless[fpr={}%]", self.fpr * 100.0)
    }
}

impl<T> BuildFilter<T> for BloomRibltHashes<T> where T: Extract {}
impl<T> Dispatcher<T> for BloomRibltHashes<T> where T: Clone + Decompose<Decomposition = T> + Extract
{}

impl<T> Algorithm<T> for BloomRibltHashes<T>
where
    T: Clone + Decompose<Decomposition = T> + Default + Extract + Measure,
{
    type Tracker = DefaultTracker;

    fn sync(&self, local: &mut T, remote: &mut T, tracker: &mut Self::Tracker) {
        assert!(
            tracker.is_ready(),
            "tracker should be ready, i.e., no captured events and not finished"
        );
        const CODED_SYMBOL_SIZE: usize =
            mem::size_of::<u64>() + mem::size_of::<u64>() + mem::size_of::<i64>();

        let hasher = RandomState::new();

        // 1. Create a bloom filter from the local join-deocompositions and send it to the remote replica.
        let local_decompositions = local.split();
        let local_filter = self.filter_from(&local_decompositions, self.fpr);

        tracker.register(DefaultEvent::LocalToRemote {
            state: 0,
            metadata: <Self as BuildFilter<T>>::size_of(&local_filter),
            upload: tracker.upload(),
        });

        // 2. Partion the remote join-decompositions into *probably* present in both replicas or
        //    *definitely not* present in the local replica.
        let (remote_common, local_unknown) = self.partition(&local_filter, remote.split());

        // 3. Build a bloom filter from the partion of *probably* common join-decompositions
        let remote_filter = self.filter_from(&remote_common, self.fpr);

        // 4. Calculate the hashes of the *probably* common join-decompositions and put them into the sketch
        //    to be streamed for synchronization
        let remote_hashes = {
            let mut remote_hashes = HashMap::new();
            let mut state = T::default();
            state.join(remote_common);
            state.split().into_iter().for_each(|d| {
                let item = d.extract();
                let item_hash = hasher.hash_one(item);

                remote_hashes.insert(item_hash, d);
            });
            remote_hashes
        };
        let mut remote_iblt = RatelessIBLT::riblt_from(remote_hashes.keys().cloned());

        // 5. Partion the local join-decompositions into *probably* present in both replicas or
        //    *definitely not* present in the remote replica. (same as 2)
        let (local_common, remote_unknown) = self.partition(&remote_filter, local_decompositions);

        // 6. Calculate the hashes of the *probably* common join-decompositions and put them into the sketch
        //    to be streamed for synchronization (same as 4)
        let local_hashes = {
            let mut local_hashes = HashMap::new();
            let mut state = T::default();
            state.join(local_common);
            state.split().into_iter().for_each(|d| {
                let item = d.extract();
                let item_hash = hasher.hash_one(item);

                local_hashes.insert(item_hash, d);
            });
            local_hashes
        };
        let mut local_iblt = RatelessIBLT::riblt_from(local_hashes.keys().cloned());

        local_iblt.find_all_differences(&mut remote_iblt);

        let sketch_size = local_iblt.sketch.len();
        assert_eq!(sketch_size, remote_iblt.sketch.len());

        //message with just received filter + sketch
        tracker.register(DefaultEvent::RemoteToLocal {
            state: local_unknown.iter().map(<T as Measure>::size_of).sum(),
            metadata: <Self as BuildFilter<T>>::size_of(&remote_filter)
                + sketch_size * CODED_SYMBOL_SIZE,
            download: tracker.download(),
        });

        let local_only_hashes_fp = local_iblt.get_local_only_symbols();
        let remote_only_hashes_fp = local_iblt.get_remote_only_symbols();

        let local_only_decompositions_fp: Vec<_> = local_only_hashes_fp
            .into_iter()
            .map(|hash| local_hashes[&hash].clone())
            .collect();

        // 7. Send remote unknown state detected using the BF
        //    Send local only state due to false positives
        //    Send remote only hashes to request for remote only state due to false positives

        tracker.register(DefaultEvent::LocalToRemote {
            state: remote_unknown
                .iter()
                .chain(&local_only_decompositions_fp)
                .map(T::size_of)
                .sum(),
            metadata: remote_only_hashes_fp.len() * mem::size_of::<u64>(),
            upload: tracker.upload(),
        });

        let remote_only_decompositions_fp: Vec<_> = remote_only_hashes_fp
            .into_iter()
            .map(|hash| remote_hashes[&hash].clone())
            .collect();

        // 8. Send remote only state due to false positives
        tracker.register(DefaultEvent::RemoteToLocal {
            state: remote_only_decompositions_fp
                .iter()
                .map(<T as Measure>::size_of)
                .sum(),
            metadata: 0,
            download: tracker.download(),
        });

        // 9. Join the appropriate join-decompositions to each replica.
        remote.join(remote_unknown);
        remote.join(local_only_decompositions_fp);

        local.join(local_unknown);
        local.join(remote_only_decompositions_fp);

        // 10. Sanity Check.
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
            let items = "a b c d e f g h i j k l"
                .split_whitespace()
                .collect::<Vec<_>>();

            for item in items {
                gset.insert(item.to_string());
            }

            gset
        };

        let mut remote = {
            let mut gset = GSet::new();
            let items = "m n o p q r s t u v w x y z"
                .split_whitespace()
                .collect::<Vec<_>>();

            for item in items {
                gset.insert(item.to_string());
            }

            gset
        };

        let (download, upload) = (Bandwidth::Kbps(0.5), Bandwidth::Kbps(0.5));
        let mut tracker = DefaultTracker::new(download, upload);
        let bloom_buckets = BloomRibltHashes::new(0.01);

        bloom_buckets.sync(&mut local, &mut remote, &mut tracker);
        assert_eq!(tracker.false_matches(), 0);

        let events = tracker.events();
        assert_eq!(events.len(), 4);
    }
}
