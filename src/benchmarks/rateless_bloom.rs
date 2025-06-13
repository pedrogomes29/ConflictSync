use std::time::Instant;

use rand::{SeedableRng, rngs::StdRng};

use crate::{
    crdt::{Decompose, Extract, Measure},
    rateless_bloom::RatelessBF,
};

use super::{awsets_with, gsets_with};

const NR_RUNS: usize = 100;

fn run_with<T>(similar: f64, local: T, remote: T)
where
    T: Clone + Decompose<Decomposition = T> + Default + Extract + Measure,
{
    println!("\n{similar}");

    let local_decompositions: Vec<_> = local.split().into_iter().map(|d| d.extract()).collect();

    let remote_decompositions: Vec<_> = remote.split().into_iter().map(|d| d.extract()).collect();

    let m_ratio = 0.5;
    let m = (local_decompositions.len() as f64 * m_ratio).ceil() as usize;
    let mut rateless_bf = RatelessBF::new(local_decompositions, m);
    for run in 0..NR_RUNS {
        rateless_bf.extend();
        let nr_positives = remote_decompositions
            .iter()
            .filter(|d| rateless_bf.contains(d))
            .count();
        println!("{run} {nr_positives}",);
    }
}

fn run_experiment<T, F>(label: &str, create_replicas: F)
where
    T: Clone + Decompose<Decomposition = T> + Default + Extract + Measure,
    F: Fn(f64) -> (T, T),
{
    let exec_time = Instant::now();
    let nr_steps = 100;
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
    run_experiment("awsets", |s| {
        let mut rng = StdRng::seed_from_u64(rand::random());
        awsets_with(20_000, s, 0.2, &mut rng)
    });
}
