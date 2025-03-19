#![allow(dead_code)]

use std::{
    env,
    fmt::Display,
    time::{Duration, Instant},
};

use crate::{
    crdt::{AWSet, GSet, Measure},
    sync::{Algorithm, baseline::Baseline, bloombuckets::BloomBuckets, buckets::Buckets},
    tracker::{Bandwidth, DefaultEvent, DefaultTracker, Telemetry},
};

use crdt::{Decompose, Extract};
use rand::{
    SeedableRng,
    distributions::{Alphanumeric, Bernoulli, DistString, Distribution, Uniform},
    rngs::StdRng,
};

mod bloom;
mod crdt;
mod riblt;
mod sync;
mod tracker;

fn gsets_with(len: usize, similar: f64, rng: &mut StdRng) -> (GSet<String>, GSet<String>) {
    assert!(
        (0.0..=1.0).contains(&similar),
        "similarity ratio should be in (0.0..=1.0)"
    );

    let sims = (len as f64 * similar) as usize;
    let diffs = len - sims;
    let dist = Uniform::new_inclusive(5, 80);

    let (mut local, mut remote) = (GSet::new(), GSet::new());

    for _ in 0..sims {
        let len = dist.sample(rng);
        let item = Alphanumeric.sample_string(rng, len);
        local.insert(item.clone());
        remote.insert(item);
    }

    for _ in 0..diffs {
        let len = dist.sample(rng);
        local.insert(Alphanumeric.sample_string(rng, len));

        let len = dist.sample(rng);
        remote.insert(Alphanumeric.sample_string(rng, len));
    }

    assert_eq!(local.len(), len);
    assert_eq!(remote.len(), len);
    assert_eq!(local.false_matches(&remote), 2 * diffs);
    (local, remote)
}

fn awsets_with(
    len: usize,
    similar: f64,
    del: f64,
    rng: &mut StdRng,
) -> (AWSet<String>, AWSet<String>) {
    assert!(
        (0.0..=1.0).contains(&similar),
        "similarity ratio should be in (0.0..=1.0)"
    );

    let sims = (len as f64 * similar) as usize;
    let diffs = len - sims;

    let dist = Uniform::new_inclusive(5, 80);
    let ratio = Bernoulli::new(del).unwrap();

    let mut common = AWSet::new();

    for _ in 0..sims {
        let len = dist.sample(rng);
        let item = Alphanumeric.sample_string(rng, len);

        if ratio.sample(rng) {
            common.insert(item.clone());
            common.remove(&item);
        } else {
            common.insert(item);
        }
    }

    let mut local = common.clone();
    let mut remote = common;

    for _ in 0..diffs {
        let len = dist.sample(rng);
        let item = Alphanumeric.sample_string(rng, len);

        if ratio.sample(rng) {
            local.insert(item.clone());
            local.remove(&item);
        } else {
            local.insert(item);
        }

        let len = dist.sample(rng);
        let item = Alphanumeric.sample_string(rng, len);

        if ratio.sample(rng) {
            remote.insert(item.clone());
            remote.remove(&item);
        } else {
            remote.insert(item);
        }
    }

    let lower_bound = (0.99 - del) * len as f64;
    let upper_bound = (1.01 - del) * len as f64;
    let one_percent_error = lower_bound..=upper_bound;

    assert!(one_percent_error.contains(&(local.len() as f64)));
    assert!(one_percent_error.contains(&(remote.len() as f64)));

    (local, remote)
}

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
    let size_of_local = <T as Measure>::size_of(&local);
    let size_of_remote = <T as Measure>::size_of(&remote);
    let avg_size_of = (size_of_local + size_of_remote) / 2;

    let links = [
        (Bandwidth::Mbps(10.0), Bandwidth::Mbps(1.0)),
        (Bandwidth::Mbps(10.0), Bandwidth::Mbps(10.0)),
        (Bandwidth::Mbps(1.0), Bandwidth::Mbps(10.0)),
    ];

    for (upload, download) in links {
        println!(
            "\n{avg_size_of} {} {}",
            upload.bits_per_sec(),
            download.bits_per_sec()
        );

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
    }
}

/// Entry point for the execution of the experiments.
///
/// The first experiment is on similar uses replica with the same cardinality and a given degree of
/// similarity. Furthermore, we repeat this experiment on different link configurations with both
/// symmetric and assymetric channels.
///
/// The second experiment is on replicas of distinct cardinalities, also with different link
/// configurations, again, with both symmetric and asymmetric channels.
fn main() {
    let exec_time = Instant::now();

    let seed = rand::random();
    let mut rng = StdRng::seed_from_u64(seed);
    eprintln!(
        "[{:.2?}] got seed with value of {seed}",
        exec_time.elapsed()
    );

    let args = env::args().collect::<Vec<_>>();
    if args.len() != 2 {
        panic!("expected an argument telling which data type to use")
    }

    let similarities = (0..=100)
        .step_by(5)
        .map(|similar| f64::from(similar) / 100.0);

    match args[1].to_lowercase().as_str() {
        "gset" => similarities.for_each(|s| {
            let (local, remote) = gsets_with(100_000, s, &mut rng);
            eprintln!(
                "[{:.2?}] gsets with similarity of {s} generated",
                exec_time.elapsed()
            );

            run_with(s, local, remote);
        }),

        // NOTE: AWSets generated with 20% of elements removed. This value is pretty conservative for
        // the particular study scenario of 15% of deleted or removed posts as in mainstream social
        // media [1].
        //
        // [1]: https://www.researchgate.net/publication/367503309_Engagement_with_fact-checked_posts_on_Reddit
        "awset" => similarities.for_each(|s| {
            let (local, remote) = awsets_with(20_000, s, 0.2, &mut rng);
            eprintln!(
                "[{:.2?}] awsets with similarity of {s} generated",
                exec_time.elapsed()
            );

            run_with(s, local, remote);
        }),
        _ => unreachable!(),
    };

    eprintln!("[{:.2?}] exiting...", exec_time.elapsed());
}
