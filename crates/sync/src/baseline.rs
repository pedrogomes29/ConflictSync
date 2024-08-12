use crdt::{Decompose, Extract};
use telemetry::{Telemetry, Tracker, TransferEvent, TransferKind};

use crate::Algorithm;

#[derive(Clone, Copy, Debug, Default)]
pub struct Baseline {}

impl<'b, R> Algorithm<R> for Baseline
where
    R: Clone + Decompose,
    for<'a> R::Decomposition<'a>: Extract,
{
    const HOPS: usize = 2;
    type Tracker = Tracker;

    fn sync(&self, alpha: &mut R, beta: &mut R, tracker: &mut Self::Tracker) {
        tracker.reset();

        let alpha_clone = alpha.clone();
        tracker.register(TransferEvent {
            state: 0,
            metadata: 0,
            kind: TransferKind::LocalToRemote,
        });

        let optimal_delta = beta.difference(&alpha_clone);
        tracker.register(TransferEvent {
            state: 0,
            metadata: 0,
            kind: TransferKind::RemoteToLocal,
        });

        alpha.join(vec![optimal_delta]);
        beta.join(vec![alpha.as_delta()]);
    }
}

#[cfg(test)]
mod tests {
    use crdt::GSet;
    use telemetry::Tracker;

    use crate::Algorithm;

    use super::Baseline;

    #[test]
    fn sync_test() {
        let mut alpha = GSet::new();
        for e in 0..=70 {
            alpha.insert(e);
        }

        let mut beta = GSet::new();
        for e in 30..=100 {
            beta.insert(e);
        }

        let baseline = Baseline {};
        let mut tracker = Tracker::new(10.0 * Tracker::MBPS, 10.0 * Tracker::MBPS);

        baseline.sync(&mut alpha, &mut beta, &mut tracker);

        // FIXME: Take into account the transmission
        assert_eq!(alpha, beta);
    }
}
