use crdt::{Decompose, Extract};
use telemetry::{Telemetry, Tracker, TransferEvent, TransferKind};

use crate::Algorithm;

#[derive(Clone, Copy, Debug, Default)]
pub struct Classic {}

impl<'b, R> Algorithm<R> for Classic
where
    R: Decompose + 'b,
    for<'a> R::Decomposition<'a>: Extract,
{
    const HOPS: usize = 2;
    type Tracker = Tracker;

    // FIXME: Take into account the transmission
    fn sync(&self, alpha: &mut R, beta: &mut R, tracker: &mut Self::Tracker) {
        tracker.reset();

        let alpha_delta = alpha.as_delta();
        tracker.register(TransferEvent {
            state: 0,
            metadata: 0,
            kind: TransferKind::LocalToRemote,
        });

        beta.join(vec![alpha_delta]);

        let beta_delta = beta.as_delta();
        tracker.register(TransferEvent {
            state: 0,
            metadata: 0,
            kind: TransferKind::RemoteToLocal,
        });

        alpha.join(vec![beta_delta]);
    }
}

#[cfg(test)]
mod tests {
    use crdt::GSet;
    use telemetry::Tracker;

    use crate::Algorithm;

    use super::Classic;

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

        let classic = Classic {};
        let mut tracker = Tracker::new(10.0 * Tracker::MBPS, 10.0 * Tracker::MBPS);

        classic.sync(&mut alpha, &mut beta, &mut tracker);

        // FIXME: Take into account the transmission
        assert_eq!(alpha, beta);
    }
}
