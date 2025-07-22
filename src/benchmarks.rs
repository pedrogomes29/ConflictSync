pub mod algorithms;

use std::collections::HashSet;

use crate::crdt::{AWSet, GSet, PNCounter};

use rand::{
    distributions::{Alphanumeric, Bernoulli, DistString, Distribution, Uniform},
    rngs::StdRng,
    Rng
};
fn 
gsets_with(len: usize, similarity: f64, rng: &mut StdRng) -> (GSet<String>, GSet<String>) {
    assert!(
        (0.0..=1.0).contains(&similarity),
        "similarity ratio should be in (0.0..=1.0)"
    );

    //derived such that sims/(sims+2*diffs) = similar
    let sims = ((2.0 * similarity * len as f64) / (1.0 + similarity)) as usize;
    let diffs = len - sims;

    let dist = Uniform::new_inclusive(5, 80);

    let mut global_seen = HashSet::new();
    let (mut local, mut remote) = (GSet::new(), GSet::new());
    
    for _ in 0..sims {
        let mut item;
        loop {
            let len = dist.sample(rng);
            item = Alphanumeric.sample_string(rng, len);
            if global_seen.insert(item.clone()) {
                break;
            }
        }
        local.insert(item.clone());
        remote.insert(item);
    }
    
    for _ in 0..diffs {
        let mut item;
        loop {
            let len = dist.sample(rng);
            item = Alphanumeric.sample_string(rng, len);
            if global_seen.insert(item.clone()) {
                break;
            }
        }
        local.insert(item);
    
        loop {
            let len = dist.sample(rng);
            item = Alphanumeric.sample_string(rng, len);
            if global_seen.insert(item.clone()) {
                break;
            }
        }
        remote.insert(item);
    }

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

fn pncounters_with(
    len: usize,
    similarity: f64,
    rng: &mut StdRng
) -> (PNCounter<String>, PNCounter<String>){
    assert!(
        (0.0..=1.0).contains(&similarity),
        "similarity ratio should be in (0.0..=1.0)"
    );

    let identical_decompositions_count = (similarity * len as f64).round() as usize;
    let dissimilar_decompositions_count = len - identical_decompositions_count;

    let id_len_dist = Uniform::new_inclusive(5, 20);
    let op_value_dist = Uniform::new_inclusive(1, 1000);

    let mut global_seen_ids = HashSet::new();
    let (mut local_counter, mut remote_counter) = (PNCounter::new(), PNCounter::new());

    for _ in 0..identical_decompositions_count {
        let mut item_id;
        loop {
            let len_id = id_len_dist.sample(rng);
            item_id = Alphanumeric.sample_string(rng, len_id);
            if global_seen_ids.insert(item_id.clone()) {
                break;
            }
        }

        let incr = op_value_dist.sample(rng);
        let decr = op_value_dist.sample(rng);

        local_counter.add(&item_id, incr);
        local_counter.sub(&item_id, decr);

        remote_counter.add(&item_id, incr);
        remote_counter.sub(&item_id, decr);
    }

    for _ in 0..dissimilar_decompositions_count {
        let mut item_id;
        loop {
            let len_id = id_len_dist.sample(rng);
            item_id = Alphanumeric.sample_string(rng, len_id);
            if global_seen_ids.insert(item_id.clone()) {
                break;
            }
        }
    
        let ((incr1,decr1),(incr2,decr2)) = loop {
            let incr1 = op_value_dist.sample(rng);
            let incr2 = op_value_dist.sample(rng);
            let decr1 = op_value_dist.sample(rng);
            let decr2 = op_value_dist.sample(rng);
        
            if incr1 != incr2 || decr1 != decr2 {
                break ((incr1,decr1),(incr2,decr2))
            }
        };
        let make_local_strictly_greater = rng.gen_bool(0.5);


        let ((local_incr,local_decr),(remote_incr,remote_decr)) = if make_local_strictly_greater {
            let local_incr = incr1.max(incr2);
            let local_decr = decr1.max(decr2);
            let remote_incr = incr1.min(incr2);
            let remote_decr = decr1.min(decr2);
            ((local_incr,local_decr),(remote_incr,remote_decr))
        } else {
            let local_incr = incr1.min(incr2);
            let local_decr = decr1.min(decr2);
            let remote_incr = incr1.max(incr2);
            let remote_decr = decr1.max(decr2);
            ((local_incr,local_decr),(remote_incr,remote_decr))
        };
    
        local_counter.add(&item_id, local_incr);
        local_counter.sub(&item_id, local_decr);

        remote_counter.add(&item_id, remote_incr);
        remote_counter.sub(&item_id, remote_decr);
      
    }

    (local_counter, remote_counter)
}