#![allow(dead_code)]
use std::time::Duration;

pub trait Telemetry {
    type Event;
    type DataCollection;

    fn reset(&mut self);
    fn register(&mut self, event: Self::Event);
    fn events(&self) -> &Vec<Self::Event>;
    fn collect(&self) -> Self::DataCollection;
}

#[derive(Clone, Copy, Debug)]
pub enum TransferKind {
    LocalToRemote,
    RemoteToLocal,
}

#[derive(Clone, Copy, Debug)]
pub struct TransferEvent {
    pub state: usize,
    pub metadata: usize,
    pub kind: TransferKind,
}

impl TransferEvent {
    /// Computes the duration of an event in seconds. The `bandwidth` is assumed to be given in
    /// bit/s. Panics if the `bandwidth` <= 0.
    pub fn duration(&self, bandwidth: f64) -> Duration {
        assert!(
            bandwidth > 0.0,
            "bandwidth should be greater than 0.0 bit/s"
        );
        Duration::from_secs_f64((self.state + self.metadata) as f64 * 8.0 / bandwidth)
    }
}

#[derive(Clone, Debug)]
pub struct Tracker {
    events: Vec<TransferEvent>,
    pub upload: f64,
    pub download: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Totals {
    pub sent: usize,
    pub state: usize,
    pub metadata: usize,
    pub duration: Duration,
}

impl Tracker {
    pub const KBPS: f64 = 1.0e3;
    pub const MBPS: f64 = 1.0e6;
    pub const GBPS: f64 = 1.0e9;

    pub const fn new(upload: f64, download: f64) -> Self {
        Self {
            events: vec![],
            upload,
            download,
        }
    }
}

impl Telemetry for Tracker {
    type Event = TransferEvent;
    type DataCollection = Totals;

    fn reset(&mut self) {
        self.events.clear()
    }

    fn register(&mut self, event: Self::Event) {
        self.events.push(event)
    }

    fn events(&self) -> &Vec<Self::Event> {
        &self.events
    }

    fn collect(&self) -> Self::DataCollection {
        Totals {
            sent: self.events.iter().map(|e| e.state + e.metadata).sum(),
            state: self.events.iter().map(|e| e.state).sum(),
            metadata: self.events.iter().map(|e| e.metadata).sum(),
            duration: self
                .events
                .iter()
                .map(|e| match e.kind {
                    TransferKind::LocalToRemote => e.duration(self.upload),
                    TransferKind::RemoteToLocal => e.duration(self.download),
                })
                .sum(),
        }
    }
}
