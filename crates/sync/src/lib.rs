use telemetry::Telemetry;

mod baseline;
mod classic;

pub use crate::baseline::Baseline;
pub use crate::classic::Classic;

/// This is the core trait that implements the different sync algorithms for state-based CRDTs. The
/// type parameter `R` denotes the type of replica that will be synced.
pub trait Algorithm<R> {
    const HOPS: usize;
    type Tracker: Telemetry;

    /// Syncs replicas `alpha` and `beta`, where `alpha` represent the replica that takes the
    /// initiative over the synchronization procedure. Upon termination, the tracker cotains the
    /// transmition events occured.
    fn sync<'a>(&self, alpha: &'a mut R, beta: &'a mut R, tracker: &mut Self::Tracker);
}
