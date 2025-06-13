use std::{cmp::{max, min}, hash::Hash};

use crate::bayesian_estimation;

use super::{RatelessBF, StoppingStrategy, StoppingStrategyFactory};


const HASH_SIZE:usize = std::mem::size_of::<u64>();
const SYMBOL_SIZE:usize = HASH_SIZE;
const COUNTER_SIZE:usize = std::mem::size_of::<u64>();
const IBLT_SYMBOL_SIZE:usize = SYMBOL_SIZE + HASH_SIZE + COUNTER_SIZE;
const RATELESS_SET_RECONCILIATION_MULTIPLIER:f64 = (1.35+1.72)/2.0;
const CONFIDENCE_LEVEL:f64 = 0.95;

const fn round_mul(multiplier_millis: usize, value: usize) -> usize {
    (multiplier_millis * value + 500) / 1000
}

const RATELESS_SET_RECONCILIATION_MULTIPLIER_MILLIS: usize = 1350;
const RATELESS_SET_RECONCILIATION_OVERHEAD: usize =
    HASH_SIZE + round_mul(RATELESS_SET_RECONCILIATION_MULTIPLIER_MILLIS, IBLT_SYMBOL_SIZE);

pub struct BayesianNoParams<T: Hash> {
    receiver_bf: RatelessBF<T>,
    alpha: usize,
    beta: usize,
}

impl<T: Hash> BayesianNoParams<T> {
    pub fn new(receiver_data: Vec<T>, m_ratio: f64) -> Self {
        let m = (receiver_data.len() as f64 * m_ratio).ceil() as usize;
        let receiver_bf = RatelessBF::new(receiver_data, m);
        Self {
            alpha: 1,
            beta: 1,
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
        self.alpha += and_ones;
        self.beta += sender_bf.m - and_ones;
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

        const SMALL_FILTER_MAX_SIZE: i32 = 2500;

        let confidence = if n_sender<SMALL_FILTER_MAX_SIZE{
            let n_receiver = self.receiver_bf.data.len();
            bayesian_estimation::numeric_posterior_tail(
                self.alpha, 
                self.alpha + self.beta, 
                sender_bf.data.len(),
                self.receiver_bf.data.len(),
                m,
                max(desired_intersection,0) as usize,
                min(n_sender as usize,n_receiver))
        }else{
            bayesian_estimation::probability_converged_beta_tail(
                self.alpha as f64,
                self.beta as f64,
                desired_intersection,
                self.receiver_bf.data.len() as i32,
                sender_bf.data.len() as i32,
                m as i32
            )
        };

        confidence > CONFIDENCE_LEVEL
    }
}