pub mod algorithms;
pub mod rateless_bloom;

use crate::crdt::{AWSet, GSet, Measure};

use rand::{
    distributions::{Alphanumeric, Bernoulli, DistString, Distribution, Uniform},
    rngs::StdRng,
};

fn gsets_with(len: usize, similarity: f64, rng: &mut StdRng) -> (GSet<String>, GSet<String>) {
    assert!(
        (0.0..=1.0).contains(&similarity),
        "similarity ratio should be in (0.0..=1.0)"
    );

    //derived such that sims/(sims+2*diffs) = similar
    let sims = ((2.0 * similarity * len as f64) / (1.0 + similarity)) as usize;
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
    similarity: f64,
    del: f64,
    rng: &mut StdRng,
) -> (AWSet<String>, AWSet<String>) {
    assert!(
        (0.0..=1.0).contains(&similarity),
        "similarity ratio should be in (0.0..=1.0)"
    );

    //derived such that sims/(sims+2*diffs) = similar
    let sims = ((2.0 * similarity * len as f64) / (1.0 + similarity)) as usize;
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
