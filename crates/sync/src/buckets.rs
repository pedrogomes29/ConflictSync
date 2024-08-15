use std::{
    hash::{BuildHasher, Hasher},
    mem,
};

use crdt::{Decompose, Difference, Extract};
use fxhash::FxBuildHasher;
use mem_sized::MemSized;
use telemetry::{Telemetry, Tracker, TransferEvent, TransferKind};

use crate::Algorithm;

#[derive(Clone, Copy, Debug)]
pub struct Dispatcher {
    load_factor: f64,
}

/// Delta buckets is a set of deltas to its corresponding hash computed from its extracted item.
type Bucket<D> = Vec<(u64, D)>;

impl Dispatcher {
    /// Divides the irredundant join-decompositions of `replica` thorugh buckets. The number of
    /// buckets is given by a load_factor which determines the expected number of decompositions in
    /// each bucket.
    fn distribute<'b, R, S>(
        &self,
        replica: &'b R,
        builder: &S,
        num_of_buckets: Option<usize>,
    ) -> Vec<Bucket<R::Decomposition<'b>>>
    where
        R: Decompose,
        for<'a> R::Decomposition<'a>: Clone + Extract,
        S: BuildHasher,
    {
        let decompositions = replica.split();
        let len =
            num_of_buckets.unwrap_or((self.load_factor * decompositions.len() as f64) as usize);

        let mut buckets = vec![vec![]; len];

        decompositions.into_iter().for_each(|d| {
            let item = d.extract().expect("decomposition must be irredundant");
            let hash = builder.hash_one(item);

            let i = hash as usize % len;
            buckets[i].push((hash, d));
        });

        // This is the first step to compute the bucket hashes under the "Merkle trick". Each
        // decomposition is sorted by its hash, which is done here.
        buckets
            .iter_mut()
            .for_each(|b| b.sort_unstable_by_key(|e| e.0));

        buckets
    }

    /// Computes the hash of bucket using the "Merkle trick". It assumes that the bucket is already
    /// sorted.
    fn hash<R, S>(bucket: &Bucket<R::Decomposition<'_>>, builder: &S) -> u64
    where
        R: Decompose,
        S: BuildHasher,
    {
        let mut singleton_hasher = builder.build_hasher();
        bucket
            .iter()
            .for_each(|(h, _)| singleton_hasher.write_u64(*h));

        singleton_hasher.finish()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Buckets {
    dispatcher: Dispatcher,
}

impl<R> Algorithm<R> for Buckets
where
    R: Default + Clone + Decompose + Difference + MemSized,
    for<'a> <R as Decompose>::Decomposition<'a>: Clone + Extract,
    for<'a> <R as Difference>::Decomposition<'a>: Into<R> + MemSized,
{
    const HOPS: usize = 3;
    type Tracker = Tracker;

    fn sync<'a>(&self, alpha: &'a mut R, beta: &'a mut R, tracker: &mut Self::Tracker) {
        tracker.reset();

        let builder = FxBuildHasher::default();

        let alpha_buckets = self.dispatcher.distribute(alpha, &builder, None);
        let alpha_bucket_hashes = alpha_buckets
            .iter()
            .map(|b| Dispatcher::hash::<R, FxBuildHasher>(&b, &builder));

        // NOTE: We also send the indices of each bucket to allow for a proper match.
        tracker.register(TransferEvent {
            state: 0,
            metadata: alpha_bucket_hashes.len() * (mem::size_of::<u64>() + mem::size_of::<usize>()),
            kind: TransferKind::LocalToRemote,
        });

        let num_of_buckets = Some(alpha_bucket_hashes.len());
        let beta_buckets = self.dispatcher.distribute(beta, &builder, num_of_buckets);
        let beta_buckets_hashes = beta_buckets
            .iter()
            .map(|b| Dispatcher::hash::<R, FxBuildHasher>(b, &builder))
            .collect::<Vec<_>>();

        let hashes = alpha_bucket_hashes
            .zip(beta_buckets_hashes)
            .collect::<Vec<_>>();

        let non_matching_at_beta = beta_buckets
            .into_iter()
            .zip(hashes.clone())
            .filter_map(|(b, (ha, hb))| if ha != hb { Some(b) } else { None })
            .map(|b| {
                let mut bucket = R::default();
                bucket.join(b.into_iter().map(|(_, d)| d).collect());
                bucket
            })
            .collect::<Vec<_>>();

        tracker.register(TransferEvent {
            state: non_matching_at_beta.iter().map(MemSized::size_of).sum(),
            metadata: non_matching_at_beta.len() * mem::size_of::<usize>(),
            kind: TransferKind::RemoteToLocal,
        });

        let non_matching_at_alpha = alpha_buckets
            .into_iter()
            .zip(hashes)
            .filter_map(|(b, (ha, hb))| if ha != hb { Some(b) } else { None })
            .map(|b| {
                let mut bucket = R::default();
                bucket.join(b.into_iter().map(|(_, d)| d).collect());
                bucket
            })
            .collect::<Vec<_>>();

        let diffs = non_matching_at_alpha
            .into_iter()
            .zip(non_matching_at_beta.clone())
            .map(|(ba, bb)| ba.difference(&bb).into())
            .collect::<Vec<_>>();

        tracker.register(TransferEvent {
            state: diffs.iter().map(MemSized::size_of).sum(),
            metadata: 0,
            kind: TransferKind::LocalToRemote,
        });

        let deltas_to_alpha = non_matching_at_beta.iter().map(|d| d.as_delta());
        alpha.join(deltas_to_alpha.collect());

        let deltas_to_beta = diffs.iter().map(|d| d.as_delta());
        beta.join(deltas_to_beta.collect());
    }
}

#[cfg(test)]
mod tests {
    use std::mem;

    use crdt::GSet;
    use telemetry::{Telemetry, Tracker};

    use crate::{
        buckets::{Buckets, Dispatcher},
        Algorithm,
    };

    #[test]
    fn sync_test() {
        let mut alpha = GSet::new();
        for e in 0..70 {
            alpha.insert(e);
        }

        let mut beta = GSet::new();
        for e in 30..100 {
            beta.insert(e);
        }

        let buckets = Buckets {
            dispatcher: Dispatcher { load_factor: 1.0 },
        };
        let mut tracker = Tracker::new(10.0 * Tracker::MBPS, 10.0 * Tracker::MBPS);

        buckets.sync(&mut alpha, &mut beta, &mut tracker);
        assert_eq!(alpha, beta);
        assert_eq!(alpha.len(), 100);

        assert_eq!(
            tracker.events().len(),
            <Buckets as Algorithm<GSet<i32>>>::HOPS
        );

        let totals = tracker.collect();
        assert_ne!(totals.state, 0);
        assert_ne!(totals.metadata, 0);
    }

    #[test]
    fn eq_sync_test() {
        let mut alpha = GSet::new();
        let mut beta = GSet::new();
        for e in 0..100 {
            alpha.insert(e);
            beta.insert(e);
        }

        let buckets = Buckets {
            dispatcher: Dispatcher { load_factor: 1.0 },
        };
        let mut tracker = Tracker::new(10.0 * Tracker::MBPS, 10.0 * Tracker::MBPS);

        assert_eq!(alpha, beta);
        buckets.sync(&mut alpha, &mut beta, &mut tracker);
        assert_eq!(alpha, beta);

        assert_eq!(
            tracker.events().len(),
            <Buckets as Algorithm<GSet<i32>>>::HOPS
        );

        dbg!(tracker.events());

        let totals = tracker.collect();
        assert_eq!(
            totals.sent,
            100 * (mem::size_of::<u64>() + mem::size_of::<usize>())
        );
        assert_eq!(totals.sent, totals.metadata);
    }
}
