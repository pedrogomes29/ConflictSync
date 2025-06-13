use std::{cmp::{max, min}, hash::Hash};

use crate::bayesian_estimation;

use super::{RatelessBF, StoppingStrategy, StoppingStrategyFactory};

const CONFIDENCE_LEVEL:f64 = 0.95;

pub struct BayesianSimilarity<T: Hash> {
    receiver_bf: RatelessBF<T>,
    alpha: usize,
    beta: usize,
    target_similarity: f64,
}

impl<T: Hash> BayesianSimilarity<T> {
    pub fn new(receiver_data: Vec<T>, target_similarity: f64, m_ratio: f64) -> Self {
        let m = (receiver_data.len() as f64 * m_ratio).ceil() as usize;
        let receiver_bf = RatelessBF::new(receiver_data, m);
        Self {
            alpha: 1,
            beta: 1,
            target_similarity,
            receiver_bf,
        }
    }
}

pub struct BayesianSimilarityFactory {
    pub m_ratio: f64,
    pub target_similarity: f64,
}

impl BayesianSimilarityFactory {
    pub fn new(m_ratio: f64, target_similarity: f64) -> Self {
        Self {
            target_similarity,
            m_ratio,
        }
    }
}

impl<T: Hash> StoppingStrategyFactory<T> for BayesianSimilarityFactory {
    type Strategy = BayesianSimilarity<T>;

    fn create(&self, elements: Vec<T>, sample_size:usize) -> Self::Strategy {
        let elements = elements.into_iter().take(sample_size).collect::<Vec<_>>();

        BayesianSimilarity::new(
            elements,
            self.target_similarity,
            self.m_ratio,
        )
    }

    fn print_name(&self) -> String {
        "Similarity".to_string()
    }
    
    fn print_params(&self) -> String {
        format!("sim={}", self.target_similarity)
    }
}

impl<T: Hash> StoppingStrategy<T> for BayesianSimilarity<T> {
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
            .count();
        
        let desired_intersection = ((self.target_similarity * self.receiver_bf.data.len() as f64)
            - true_negatives as f64)
            .round() as i32;

            const SMALL_FILTER_MAX_SIZE: usize = 2500;

            let confidence = if sender_bf.data.len()<SMALL_FILTER_MAX_SIZE{
                let n_receiver = self.receiver_bf.data.len();
                bayesian_estimation::numeric_posterior_tail(
                    self.alpha, 
                    self.alpha + self.beta, 
                    sender_bf.data.len(),
                    self.receiver_bf.data.len(),
                    self.receiver_bf.m,
                    max(desired_intersection,0) as usize,
                    min(sender_bf.data.len(),n_receiver))
            }else{
                bayesian_estimation::probability_converged_beta_tail(
                    self.alpha as f64,
                    self.beta as f64,
                    desired_intersection,
                    self.receiver_bf.data.len() as i32,
                    sender_bf.data.len() as i32,
                    self.receiver_bf.m as i32
                )
            };

        confidence > CONFIDENCE_LEVEL
    }
}