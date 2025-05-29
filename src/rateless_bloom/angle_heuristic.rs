use std::{collections::VecDeque, f64::consts::PI, hash::Hash};
use super::{RatelessBF, StoppingStrategy, StoppingStrategyFactory};

pub struct AngleHeuristic<T> {
    elements: Vec<T>,
    angle_threshold_deg: f64,
    window_size: usize,
    recent_angles: VecDeque<f64>,
    last_normalized: Option<f64>,
}

impl<T: Hash> AngleHeuristic<T> {
    pub fn new(elements: Vec<T>, angle_threshold_deg: f64, window_size: usize) -> Self {
        Self {
            elements,
            angle_threshold_deg,
            window_size,
            recent_angles: VecDeque::with_capacity(window_size),
            last_normalized: None,
        }
    }
}

pub struct AngleHeuristicFactory {
    angle_threshold_deg: f64,
    window_size: usize,
}

impl AngleHeuristicFactory {
    pub fn new(angle_threshold_deg: f64, window_size: usize) -> Self {
        Self {
            angle_threshold_deg,
            window_size,
        }
    }
}

impl<T: Hash> StoppingStrategyFactory<T> for AngleHeuristicFactory {
    type Strategy = AngleHeuristic<T>;

    fn create(&self, elements: Vec<T>, _sample_size: usize) -> Self::Strategy {
        AngleHeuristic::new(elements, self.angle_threshold_deg, self.window_size)
    }

    fn print_name(&self) -> String {
        "Heuristic".to_string()
    }
    
    fn print_params(&self) -> String {
        format!("angle={}", self.angle_threshold_deg)
    }
}

impl<T: Hash> StoppingStrategy<T> for AngleHeuristic<T> {
    fn on_extend(&mut self, bf: &RatelessBF<T>) {
        let (positives, negatives): (Vec<_>, Vec<_>) = self
            .elements
            .drain(..)
            .partition(|e| bf.contains(e));

        let normalized = positives.len() as f64 / (positives.len() + negatives.len()).max(1) as f64;

        if let Some(prev) = self.last_normalized {
            let dy = normalized - prev;
            let angle = dy.abs().atan() * 180.0 / PI;

            if self.recent_angles.len() == self.window_size {
                self.recent_angles.pop_front();
            }
            self.recent_angles.push_back(angle);
        }

        self.last_normalized = Some(normalized);
        self.elements = positives.into_iter().chain(negatives).collect();
    }

    fn should_stop(&mut self, _: &RatelessBF<T>) -> bool {
        if self.recent_angles.len() == self.window_size {
            let avg: f64 = self.recent_angles.iter().sum::<f64>() / self.window_size as f64;
            return avg < self.angle_threshold_deg
        }
        false
    }
}

