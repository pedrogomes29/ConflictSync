use crdt::Decompose;
use mem_sized::MemSized;
use telemetry::{Telemetry, Tracker, TransferEvent, TransferKind};

use crate::Algorithm;

#[derive(Clone, Copy, Debug, Default)]
pub struct Classic {}

impl<R> Algorithm<R> for Classic
where
    R: Decompose + MemSized,
{
    const HOPS: usize = 2;
    type Tracker = Tracker;

    fn sync<'a>(&self, alpha: &'a mut R, beta: &'a mut R, tracker: &mut Self::Tracker) {
        tracker.reset();

        let alpha_delta = alpha.as_delta();
        tracker.register(TransferEvent {
            state: alpha.size_of(),
            metadata: 0,
            kind: TransferKind::LocalToRemote,
        });

        beta.join(vec![alpha_delta]);

        let beta_delta = beta.as_delta();
        tracker.register(TransferEvent {
            state: beta.size_of(),
            metadata: 0,
            kind: TransferKind::RemoteToLocal,
        });

        alpha.join(vec![beta_delta]);
    }
}

#[cfg(test)]
mod tests {
    use std::mem;

    use crdt::GSet;
    use telemetry::{Telemetry, Tracker};

    use crate::Algorithm;

    use super::Classic;

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

        let classic = Classic {};
        let mut tracker = Tracker::new(10.0 * Tracker::MBPS, 10.0 * Tracker::MBPS);

        classic.sync(&mut alpha, &mut beta, &mut tracker);
        assert_eq!(alpha, beta);
        assert_eq!(alpha.len(), 100);

        assert_eq!(
            tracker.events().len(),
            <Classic as Algorithm<GSet<i32>>>::HOPS
        );

        let totals = tracker.collect();
        assert_eq!(totals.sent, (70 + 100) * mem::size_of::<i32>());
        assert_eq!(totals.metadata, 0);
    }
}
