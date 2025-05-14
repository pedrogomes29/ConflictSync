#![allow(dead_code)]

use std::env;

mod benchmarks;
mod bloom;
mod crdt;
mod rateless_bloom;
mod riblt;
mod sync;
mod tracker;

/// Entry point for the execution of the experiments.
///
/// The first experiment is on similar uses replica with the same cardinality and a given degree of
/// similarity. Furthermore, we repeat this experiment on different link configurations with both
/// symmetric and assymetric channels.
///
/// The second experiment is on replicas of distinct cardinalities, also with different link
/// configurations, again, with both symmetric and asymmetric channels.
///
/// The third experiment tests rateless bloom filters (TODO: better explanation)
fn main() {
    let args = env::args().collect::<Vec<_>>();
    if args.len() != 2 {
        panic!("expected an argument telling which data type to use")
    }

    match args[1].to_lowercase().as_str() {
        "gset" => benchmarks::algorithms::run_gset_experiment(),

        // NOTE: AWSets generated with 20% of elements removed. This value is pretty conservative for
        // the particular study scenario of 15% of deleted or removed posts as in mainstream social
        // media [1].
        //
        // [1]: https://www.researchgate.net/publication/367503309_Engagement_with_fact-checked_posts_on_Reddit
        "awset" => benchmarks::algorithms::run_awset_experiment(),

        "ratelessbf" => benchmarks::rateless_bloom::run_gset_experiment(),

        _ => unreachable!(),
    };
}
