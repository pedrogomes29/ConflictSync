use crdt::{Decompose, MemSized};
use telemetry::{Telemetry, Tracker, TransferEvent, TransferKind};

use crate::Algorithm;

#[derive(Clone, Copy, Debug, Default)]
pub struct Baseline {}

impl<R> Algorithm<R> for Baseline
where
    R: Default + Decompose + MemSized,
{
    const HOPS: usize = 2;
    type Tracker = Tracker;

    fn sync(&self, alpha: &mut R, beta: &mut R, tracker: &mut Self::Tracker) {
        tracker.reset();

        tracker.register(TransferEvent {
            state: alpha.size_of(),
            metadata: 0,
            kind: TransferKind::LocalToRemote,
        });

        let optimal_diff = {
            let mut replica = R::default();
            replica.join(vec![beta.difference(alpha)]);
            replica
        };
        tracker.register(TransferEvent {
            state: optimal_diff.size_of(),
            metadata: 0,
            kind: TransferKind::RemoteToLocal,
        });

        alpha.join(vec![optimal_diff.as_delta()]);
        beta.join(vec![alpha.as_delta()]);
    }
}

#[cfg(test)]
mod tests {
    use std::mem;

    use crdt::GSet;
    use telemetry::{Telemetry, Tracker};

    use crate::Algorithm;

    use super::Baseline;

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

        let baseline = Baseline {};
        let mut tracker = Tracker::new(10.0 * Tracker::MBPS, 10.0 * Tracker::MBPS);

        baseline.sync(&mut alpha, &mut beta, &mut tracker);
        assert_eq!(alpha, beta);
        assert_eq!(alpha.len(), 100);

        assert_eq!(
            tracker.events().len(),
            <Baseline as Algorithm<GSet<i32>>>::HOPS
        );

        let totals = tracker.collect();
        assert_eq!(totals.sent, (70 + 30) * mem::size_of::<i32>());
        assert_eq!(totals.metadata, 0);
    }
}
