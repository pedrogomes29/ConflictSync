#![allow(dead_code)]

use std::{
    f64::consts::LN_2,
    fmt::Display,
    time::{Duration, Instant},
};

use crate::{
    benchmarks::{awsets_with, gsets_with},
    crdt::{Decompose, Extract, Measure},
    sync::{
        Algorithm, baseline::Baseline, bloombuckets::BloomBuckets,
        bloomribltbuckets::BloomRibltBuckets, bloomriblthashes::BloomRibltHashes, buckets::Buckets,
        bucketsriblt::RibltBuckets, rbloomriblthashes_heuristic::RBloomRibltHashesHeuristic,
        rbloomriblthashes_similarity::RBloomRibltHashesSimilarity, riblthashes::RibltHashes,
    },
    tracker::{Bandwidth, DefaultEvent, DefaultTracker, Telemetry},
};

use rand::{SeedableRng, rngs::StdRng};

type Replica<T> = (T, Bandwidth);

/// Runs the specified protocol and outputs the metrics obtained.
fn run<T, A>(algo: &A, similar: f64, local: Replica<T>, remote: Replica<T>)
where
    T: Clone + Decompose<Decomposition = T> + Default + Extract + Measure,
    A: Algorithm<T, Tracker = DefaultTracker> + Display,
{
    assert!(
        (0.0..=1.0).contains(&similar),
        "similarity should be a ratio between 0.0 and 1.0"
    );

    let (mut local, upload) = local;
    let (mut remote, download) = remote;

    let mut tracker = DefaultTracker::new(download, upload);
    algo.sync(&mut local, &mut remote, &mut tracker);

    let diffs = tracker.false_matches();
    if diffs > 0 {
        eprintln!("{algo} not totally synced with {diffs} false matches");
    }

    let events = tracker.events();
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

fn run_with<T>(similar: f64, local: T, remote: T)
where
    T: Clone + Decompose<Decomposition = T> + Default + Extract + Measure,
{
    let local_only_size = <T as Measure>::size_of(&local.difference(&remote));
    let remote_only_size = <T as Measure>::size_of(&remote.difference(&local));

    let theoretical_minimum = local_only_size + remote_only_size;

    let links = [
        //(Bandwidth::Mbps(10.0), Bandwidth::Mbps(1.0)),
        (Bandwidth::Mbps(10.0), Bandwidth::Mbps(10.0)),
        //(Bandwidth::Mbps(1.0), Bandwidth::Mbps(10.0)),
    ];

    for (upload, download) in links {
        println!(
            "\n{theoretical_minimum} {} {}",
            upload.bits_per_sec(),
            download.bits_per_sec()
        );

        /*
        let algo = Baseline::new();
        run(
            &algo,
            similar,
            (local.clone(), upload),
            (remote.clone(), download),
        );

        for lf in [0.2, 1.0, 5.0] {
            let algo = Buckets::new(lf);
            run(
                &algo,
                similar,
                (local.clone(), upload),
                (remote.clone(), download),
            );
        }

        for lf in [0.2, 1.0, 5.0] {
            let algo = RibltBuckets::new(lf);
            run(
                &algo,
                similar,
                (local.clone(), upload),
                (remote.clone(), download),
            );
        }

        let algo = RibltHashes::new();
        run(
            &algo,
            similar,
            (local.clone(), upload),
            (remote.clone(), download),
        );

        for fpr in [0.01, 0.25] {
            for lf in [1.0, 0.2] {
                let algo = BloomBuckets::new(fpr, lf);
                run(
                    &algo,
                    similar,
                    (local.clone(), upload),
                    (remote.clone(), download),
                );
            }
        }

        for fpr in [0.01, 0.25] {
            for lf in [1.0, 0.2] {
                let algo = BloomRibltBuckets::new(fpr, lf);
                run(
                    &algo,
                    similar,
                    (local.clone(), upload),
                    (remote.clone(), download),
                );
            }
        }
        */

        /*
        let fprs: Vec<f64> = (1..=500).map(|i| i as f64 / 1000.0).collect();

        for fpr in fprs {
            let algo = BloomRibltHashes::new(fpr);
            run(
                &algo,
                similar,
                (local.clone(), upload),
                (remote.clone(), download),
            );
        }
        */

        for m_ratio in [1.0] {
            for angle_threshold_deg in [0.2] {
                let algo = RBloomRibltHashesHeuristic::new(m_ratio, angle_threshold_deg);
                run(
                    &algo,
                    similar,
                    (local.clone(), upload),
                    (remote.clone(), download),
                );
            }
        }

        for m_ratio in [1.0] {
            for similarity in [0.95] {
                let algo = RBloomRibltHashesSimilarity::new(m_ratio, similarity);
                run(
                    &algo,
                    similar,
                    (local.clone(), upload),
                    (remote.clone(), download),
                );
            }
        }
    }
}

fn run_experiment<T, F>(label: &str, create_replicas: F)
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
        let (local, remote) = create_replicas(s);
        eprintln!(
            "[{:.2?}] {label} with similarity {s} generated",
            exec_time.elapsed()
        );
        run_with(s, local, remote);
    }

    eprintln!("[{:.2?}] exiting...", exec_time.elapsed());
}

pub fn run_gset_experiment() {
    run_experiment("gsets", |s| {
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
    run_experiment("awsets", |s| {
        let mut rng = StdRng::seed_from_u64(rand::random());
        awsets_with(20_000, s, 0.2, &mut rng)
    });
}
