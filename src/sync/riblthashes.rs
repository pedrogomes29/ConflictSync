use std::{collections::HashMap, fmt::Display, hash::RandomState, marker::PhantomData, mem};

use crate::{
    crdt::{Decompose, Extract, Measure},
    riblt::{RatelessIBLT, Symbol},
    tracker::{DefaultEvent, DefaultTracker, Telemetry},
};

use std::hash::BuildHasher;

use super::{Algorithm, BuildRatelessIBLT};

pub struct RibltHashes<T> {
    _marker: PhantomData<T>,
}

impl<T> RibltHashes<T> {
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T> Display for RibltHashes<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RIBLT+Hashing")
    }
}

impl<T> BuildRatelessIBLT<T> for RibltHashes<T> where T: Symbol {}

impl<T> Algorithm<T> for RibltHashes<T>
where
    T: Clone + Extract + Decompose<Decomposition = T> + Measure,
{
    type Tracker = DefaultTracker;

    fn sync(&self, local: &mut T, remote: &mut T, tracker: &mut Self::Tracker) {
        const CODED_SYMBOL_SIZE: usize =
            mem::size_of::<u64>() + mem::size_of::<u64>() + mem::size_of::<i64>();

        assert!(
            tracker.is_ready(),
            "tracker should be ready, i.e., no captured events and not finished"
        );

        // 1. Create a rateless IBLT from the hash of the local join-deocompositions and send it
        //    to the remote replica.
        let hasher = RandomState::new();
        let mut local_hashes = HashMap::new();

        local.split().into_iter().for_each(|d| {
            let item = d.extract();
            let item_hash = hasher.hash_one(item);

            local_hashes.insert(item_hash, d);
        });
        let mut local_iblt = RatelessIBLT::riblt_from(local_hashes.keys().cloned());

        // 2. Repeat the procedure from 1., but now on the remote replica.
        let mut remote_hashes = HashMap::new();

        remote.split().into_iter().for_each(|d| {
            let item = d.extract();
            let item_hash = hasher.hash_one(item);

            remote_hashes.insert(item_hash, d);
        });
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

        let remote_only_hashes = remote_iblt.get_local_only_symbols();
        let local_only_hashes = remote_iblt.get_remote_only_symbols();

        let remote_only_decompositions: Vec<_> = remote_only_hashes
            .into_iter()
            .map(|hash| remote_hashes[&hash].clone())
            .collect();

        // 4. Send remote only state corresponding to remote only hashes,
        //    Send local only hashes to request for local only state
        tracker.register(DefaultEvent::RemoteToLocal {
            state: remote_only_decompositions
                .iter()
                .map(<T as Measure>::size_of)
                .sum(),
            metadata: local_only_hashes.iter().count() * mem::size_of::<u64>(),
            download: tracker.download(),
        });

        let local_only_decompositions: Vec<_> = local_only_hashes
            .into_iter()
            .map(|hash| local_hashes[&hash].clone())
            .collect();

        // 5. Send local only state corresponding to local only hashes,
        tracker.register(DefaultEvent::LocalToRemote {
            state: local_only_decompositions
                .iter()
                .map(<T as Measure>::size_of)
                .sum(),
            metadata: 0,
            upload: tracker.upload(),
        });

        // 5. Join the appropriate join-decompositions to each replica.
        local.join(remote_only_decompositions);
        remote.join(local_only_decompositions);

        // 6. Sanity check.
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
        let buckets = RibltHashes::new();

        buckets.sync(&mut local, &mut remote, &mut tracker);

        assert_eq!(tracker.false_matches(), 0);
    }
}
