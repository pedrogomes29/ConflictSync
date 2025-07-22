#![allow(dead_code)]

use std::{
    fmt::Display, time::{Duration, Instant}
};

use crate::{
    benchmarks::{awsets_with, gsets_with, pncounters_with}, crdt::{Decompose, Extract, Measure}, sync::{
        baseline::Baseline, bloombuckets::BloomBuckets, bloomriblthashes::BloomRibltHashes, buckets::Buckets, riblthashes::RibltHashes, Algorithm
    }, tracker::{Bandwidth, DefaultEvent, DefaultTracker, Telemetry}
};

use rand::{SeedableRng, rngs::StdRng};

const NR_TRIALS:usize = 1;

type Replica<T> = (T, Bandwidth);

//creates a vector of events corresponding to the average event for each message over multiple experiments
fn average_tracker_events(
    message_to_events: Vec<Vec<DefaultEvent>>,
    nr_experiments: usize,
    upload: Bandwidth,
    download: Bandwidth,
) -> Vec<DefaultEvent> {
    message_to_events
        .into_iter()
        .map(|message_events| {
            let (total_state, total_metadata): (usize, usize) = message_events.iter().fold((0, 0), |(s, m), e| {
                let (event_state, event_metadata) = match e {
                    DefaultEvent::LocalToRemote { state, metadata, .. }
                    | DefaultEvent::RemoteToLocal { state, metadata, .. } => (*state, *metadata),
                };
                (s + event_state, m + event_metadata)
            });

            let avg_state = total_state / nr_experiments;
            let avg_metadata = total_metadata / nr_experiments;

            match message_events.first().expect("Expected at least one event") {
                DefaultEvent::LocalToRemote { .. } => {
                    DefaultEvent::LocalToRemote { state: avg_state, metadata: avg_metadata, upload }
                }
                DefaultEvent::RemoteToLocal { .. } => {
                    DefaultEvent::RemoteToLocal { state: avg_state, metadata: avg_metadata, download }
                }
            }
        })
        .collect()
}


/// Runs the specified protocol and outputs the metrics obtained.
fn run<T, A>(algo: &A, similar: f64, local: Replica<T>, remote: Replica<T>) -> DefaultTracker
where
    T: Clone + Decompose<Decomposition = T> + Default + Extract + Measure,
    A: Algorithm<T, Tracker = DefaultTracker> + Display,
{
    assert!(
        (0.0..=1.0).contains(&similar),
        "similarity should be a ratio between 0.0 and 1.0"
    );

    //eprintln!("{algo}");

    let (mut local, upload) = local;
    let (mut remote, download) = remote;

    let mut tracker = DefaultTracker::new(download, upload);
    algo.sync(&mut local, &mut remote, &mut tracker);

    let diffs = tracker.false_matches();
    if diffs > 0 {
        panic!("{algo} not totally synced with {diffs} false matches");
    }

    tracker
}

fn run_trial<T,A>(algo: &A, similar: f64, replicas: Vec<(T,T)>, upload:Bandwidth, download:Bandwidth)
where
    T: Clone + Decompose<Decomposition = T> + Default + Extract + Measure,
    A: Algorithm<T, Tracker = DefaultTracker> + Display,
{
    let nr_experiments = replicas.len();
    let message_to_events = replicas.into_iter().enumerate().fold(
        Vec::<Vec<DefaultEvent>>::new(),
        |mut acc, (_trial_nr, (local, remote))| {
            let tracker = run(
                algo,
                similar,
                (local, upload),
                (remote, download),
            );
            for (idx, event) in tracker.events().iter().cloned().enumerate() {
                if idx == acc.len() {
                    acc.push(Vec::new());
                }
                acc[idx].push(event);
            }
            acc
        },
    );
    
    let events = average_tracker_events(message_to_events, nr_experiments, upload, download);

    println!(
        "{algo} {} {} {:.3}",
        events.iter().map(DefaultEvent::state).sum::<usize>(),
        events.iter().map(DefaultEvent::metadata).sum::<usize>(),
        events
            .iter()
            .filter_map(|e| e.duration().ok())
            .sum::<Duration>()
            .as_secs_f64(),
    );

}



fn run_with<T>(similar: f64, replicas: Vec<(T,T)>)
where
    T: Clone + Decompose<Decomposition = T> + Default + Extract + Measure,
{   

    let theoretical_minimum: usize = replicas.iter().map(|(local, remote)|-> usize{
        let local_only_size = <T as Measure>::size_of(&local.difference(&remote));
        let remote_only_size = <T as Measure>::size_of(&remote.difference(&local));
        local_only_size + remote_only_size
    }).sum::<usize>()/replicas.len();

    let links = [
        (Bandwidth::Mbps(10.0), Bandwidth::Mbps(1.0)),
        (Bandwidth::Mbps(10.0), Bandwidth::Mbps(10.0)),
        (Bandwidth::Mbps(1.0), Bandwidth::Mbps(10.0)),
    ];

    for (upload, download) in links {
        println!(
            "\n{theoretical_minimum} {} {}",
            upload.bits_per_sec(),
            download.bits_per_sec()
        );

        
        let algo = Baseline::new();
        run_trial(
            &algo,
            similar,
            replicas.clone(),
            upload,
            download
        );

        for lf in [0.2, 1.0, 5.0] {
            let algo = Buckets::new(lf);
            run_trial(
                &algo,
                similar,
                replicas.clone(),
                upload,
                download
            );
        }
        
        let algo = RibltHashes::new();
        run_trial(
            &algo,
            similar,
            replicas.clone(),
            upload,
            download
        );
              
        for fpr in [0.01, 0.25] {
            for lf in [1.0, 0.2] {
                let algo = BloomBuckets::new(fpr, lf);
                run_trial(
                    &algo,
                    similar,
                    replicas.clone(),
                    upload,
                    download
                );
            }
        }
        

        for fpr in [0.01, 0.1, 0.25]  {
            let algo = BloomRibltHashes::new(fpr);
            run_trial(
                &algo,
                similar,
                replicas.clone(),
                upload,
                download
            );
        }
    }
}

fn run_experiment<T, F>(label: &str, nr_trials: usize, create_replicas: F)
where
    T: Clone + Decompose<Decomposition = T> + Default + Extract + Measure,
    F: Fn(f64) -> (T, T),
{
    let exec_time = Instant::now();
    let nr_steps = 20;
    let start_similarity = 0;
    let end_similarity = 100;
    let step = ((end_similarity - start_similarity) as f64) / nr_steps as f64;

    let similarities = (0..=nr_steps)
        .map(|i| start_similarity as f64 + i as f64 * step)
        .map(|val| val / 100.0);

    println!("{start_similarity} {end_similarity} {nr_steps}");

    for s in similarities {
        let replicas: Vec<_>= (0..nr_trials).map(|_|create_replicas(s)).collect();
        eprintln!(
            "[{:.2?}] {label} with similarity {s} generated",
            exec_time.elapsed()
        );
        run_with(s, replicas);
    }

    eprintln!("[{:.2?}] exiting...", exec_time.elapsed());
}

pub fn run_gset_experiment() {
    run_experiment("gsets", NR_TRIALS, |s| {
        let mut rng = StdRng::seed_from_u64(rand::random());
        gsets_with(100_000, s, &mut rng)
    });
}

pub fn run_awset_experiment() {
    // NOTE: AWSets generated with 20% of elements removed. This value is pretty conservative for
    // the particular study scenario of 15% of deleted or removed posts as in mainstream social
    // media [1].
    //
    // [1]: https://www.researchgate.net/publication/367503309_Engagement_with_fact-checked_posts_on_Reddit
    run_experiment("awsets", NR_TRIALS, |s| {
        let mut rng = StdRng::seed_from_u64(rand::random());
        awsets_with(20_000, s, 0.2, &mut rng)
    });
}

pub fn run_pncounter_experiment() {
    run_experiment("pncounters", NR_TRIALS, |s| {
        let mut rng = StdRng::seed_from_u64(rand::random());
        pncounters_with(100_000, s, &mut rng)
    });
}
