use std::time::Duration;

pub trait Telemetry {
    type Event;

    fn is_ready(&self) -> bool;
    fn register(&mut self, event: Self::Event);
    fn events(&self) -> &Vec<Self::Event>;
    fn finish(&mut self, false_matches: usize);
    fn false_matches(&self) -> usize;
}

/// Network Bandwidth value in bits per second.
#[derive(Clone, Copy, Debug)]
pub enum Bandwidth {
    Kbps(f64),
    Mbps(f64),
    Gbps(f64),
}

impl Bandwidth {
    pub fn bits_per_sec(&self) -> f64 {
        match self {
            Bandwidth::Kbps(b) => b * 1.0e3,
            Bandwidth::Mbps(b) => b * 1.0e6,
            Bandwidth::Gbps(b) => b * 1.0e9,
        }
    }

    pub fn bytes_per_sec(&self) -> f64 {
        self.bits_per_sec() / 8.0
    }
}

/// Type of Event used by the [`DefaultTracker`].
/// It holds the size of the transfered payload in Bytes and estimates the duration based on the
/// bandwidth provided by the tracker that registers these kind of events.
#[derive(Debug,Clone)]
pub enum DefaultEvent {
    LocalToRemote {
        state: usize,
        metadata: usize,
        upload: Bandwidth,
    },
    RemoteToLocal {
        state: usize,
        metadata: usize,
        download: Bandwidth,
    },
}

impl DefaultEvent {
    #[inline]
    pub const fn state(&self) -> usize {
        match &self {
            Self::LocalToRemote { state, .. } => *state,
            Self::RemoteToLocal { state, .. } => *state,
        }
    }

    #[inline]
    pub const fn metadata(&self) -> usize {
        match &self {
            Self::LocalToRemote { metadata, .. } => *metadata,
            Self::RemoteToLocal { metadata, .. } => *metadata,
        }
    }

    #[inline]
    pub const fn bytes(&self) -> usize {
        self.state() + self.metadata()
    }

    #[inline]
    pub const fn upload(&self) -> Bandwidth {
        match &self {
            Self::LocalToRemote { upload, .. } => *upload,
            _ => unreachable!(),
        }
    }

    #[inline]
    pub const fn download(&self) -> Bandwidth {
        match &self {
            Self::RemoteToLocal { download, .. } => *download,
            _ => unreachable!(),
        }
    }

    #[inline]
    pub fn duration(&self) -> Result<Duration, Duration> {
        let bandwidth = match &self {
            Self::LocalToRemote { upload, .. } => *upload,
            Self::RemoteToLocal { download, .. } => *download,
        }
        .bytes_per_sec();

        if bandwidth > 0.0 {
            Ok(Duration::from_secs_f64(self.bytes() as f64 / bandwidth))
        } else {
            Err(Duration::ZERO)
        }
    }
}

/// Default [`Tracker`] for operations over the Network.
#[derive(Debug)]
pub struct DefaultTracker {
    events: Vec<DefaultEvent>,
    diffs: Option<usize>,
    download: Bandwidth,
    upload: Bandwidth,
}

impl DefaultTracker {
    #[inline]
    #[must_use]
    pub fn new(download: Bandwidth, upload: Bandwidth) -> Self {
        Self {
            events: vec![],
            diffs: None,
            download,
            upload,
        }
    }
}

impl DefaultTracker {
    #[inline]
    pub const fn download(&self) -> Bandwidth {
        self.download
    }

    #[inline]
    pub const fn upload(&self) -> Bandwidth {
        self.upload
    }
}

impl Telemetry for DefaultTracker {
    type Event = DefaultEvent;

    fn is_ready(&self) -> bool {
        self.events.is_empty() && self.diffs.is_none()
    }

    fn register(&mut self, event: Self::Event) {
        if self.diffs.is_none() {
            self.events.push(event);
        }
    }

    fn events(&self) -> &Vec<Self::Event> {
        &self.events
    }

    fn finish(&mut self, diffs: usize) {
        if self.diffs.is_none() {
            self.diffs = Some(diffs)
        }
    }

    fn false_matches(&self) -> usize {
        self.diffs
            .expect("`finish()` should be called before `diffs()`")
    }
}
