use std::hash::Hash;

use statrs::distribution::{Beta, ContinuousCDF};

use super::{RatelessBF, StoppingStrategy, StoppingStrategyFactory};


const HASH_SIZE:usize = std::mem::size_of::<u64>();
const SYMBOL_SIZE:usize = HASH_SIZE;
const COUNTER_SIZE:usize = std::mem::size_of::<u64>();
const IBLT_SYMBOL_SIZE:usize = SYMBOL_SIZE + HASH_SIZE + COUNTER_SIZE;
const RATELESS_SET_RECONCILIATION_MULTIPLIER:f64 = (1.35+1.72)/2.0;

const fn round_mul(multiplier_millis: usize, value: usize) -> usize {
    (multiplier_millis * value + 500) / 1000
}

const RATELESS_SET_RECONCILIATION_MULTIPLIER_MILLIS: usize = 1350;
const RATELESS_SET_RECONCILIATION_OVERHEAD: usize =
    HASH_SIZE + round_mul(RATELESS_SET_RECONCILIATION_MULTIPLIER_MILLIS, IBLT_SYMBOL_SIZE);

pub struct BayesianNoParams<T: Hash> {
    receiver_bf: RatelessBF<T>,
    alpha: f64,
    beta: f64,
}

impl<T: Hash> BayesianNoParams<T> {
    pub fn new(receiver_data: Vec<T>, m_ratio: f64) -> Self {
        let m = (receiver_data.len() as f64 * m_ratio).ceil() as usize;
        let receiver_bf = RatelessBF::new(receiver_data, m);
        Self {
            alpha: 1.0,
            beta: 1.0,
            receiver_bf,
        }
    }
}

pub struct BayesianNoParamsFactory {
    pub m_ratio: f64,
}

impl BayesianNoParamsFactory {
    pub fn new(m_ratio: f64) -> Self {
        Self {
            m_ratio,
        }
    }
}

impl<T: Hash> StoppingStrategyFactory<T> for BayesianNoParamsFactory {
    type Strategy = BayesianNoParams<T>;

    fn create(&self, elements: Vec<T>, sample_size:usize) -> Self::Strategy {
        let elements = elements.into_iter().take(sample_size).collect::<Vec<_>>();

        BayesianNoParams::new(
            elements,
            self.m_ratio,
        )
    }

    fn print_name(&self) -> String {
        "NoParams".to_string()
    }
    
    fn print_params(&self) -> String {
        "".to_string()
    }
}

impl<T: Hash> StoppingStrategy<T> for BayesianNoParams<T> {
    fn on_extend(&mut self, sender_bf: &RatelessBF<T>) {
        let last_sender_slice = sender_bf.bloom_filters.last().unwrap();
        self.receiver_bf.extend_with_hashers(last_sender_slice.hashers());

        let receiver_last_slice = self.receiver_bf.bloom_filters.last().unwrap();
        let mut tmp = last_sender_slice.bitslice().to_bitvec();
        tmp &= receiver_last_slice.bitslice();

        let and_ones = tmp.count_ones();
        self.alpha += and_ones as f64;
        self.beta += (sender_bf.m - and_ones) as f64;
    }

    fn should_stop(&mut self, sender_bf: &RatelessBF<T>) -> bool {
        let true_negatives = self
            .receiver_bf
            .data
            .iter()
            .filter(|e| !sender_bf.contains(e))
            .count() as i32;
        
        let n_sender = sender_bf.data.len() as i32;
        let m = sender_bf.m;
        let m_bytes = m/8;
        let fpr = 1.0 - (1.0 - 1.0/m as f64).powi(n_sender);
        let desired_new_negatives = m_bytes/RATELESS_SET_RECONCILIATION_OVERHEAD;
        let desired_false_positives = (desired_new_negatives as f64/(1.0-fpr)).round() as i32;
        let desired_intersection = n_sender - true_negatives - desired_false_positives;

        let confidence = probability_converged_beta_tail(
            self.alpha,
            self.beta,
            desired_intersection,
            self.receiver_bf.data.len() as i32,
            sender_bf.data.len() as i32,
            sender_bf.m as i32,
        );

        confidence > 0.90
    }
}


fn estimate_intersection(
    observed_inner_product: f64,
    n_receiver: i32,
    n_sender: i32,
    k: f64,
    m: f64,
) -> i32 {
    let y = 1.0 - 1.0 / m;

    let numerator = observed_inner_product / (k * m) + y.powi(n_receiver) + y.powi(n_sender) - 1.0;

    n_receiver + n_sender - (numerator.ln() / y.ln()).round() as i32
}

fn probability_converged_beta_tail(
    alpha: f64,
    beta: f64,
    desired_intersection: i32,
    n_receiver: i32,
    n_sender: i32,
    m: i32,
) -> f64 {
    let y = 1.0 - 1.0 / m as f64;

    let theta_target =
        1.0 - y.powi(n_sender) - y.powi(n_receiver) + y.powi(n_sender + n_receiver - desired_intersection);


    if let Ok(beta_dist) = Beta::new(alpha, beta) {
        1.0 - beta_dist.cdf(theta_target)
    } else {
        0.0 // fallback: treat as zero confidence if the distribution fails
    }
}
 