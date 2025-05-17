use super::bloom::BloomFilter;
use std::{
    collections::VecDeque,
    error::Error,
    f64::consts::PI,
    fmt::{self, Display, Formatter},
    hash::Hash,
    mem,
};

#[derive(Debug)]
struct ConvergenceError(String);

impl Display for ConvergenceError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for ConvergenceError {}

pub struct RatelessBF<T: Hash> {
    bloom_filters: Vec<BloomFilter<T>>,
    data: Vec<T>,
    m: usize,
}

impl<T> RatelessBF<T>
where
    T: Hash,
{
    #[inline]
    #[must_use]
    pub fn new(data: Vec<T>, m: usize) -> Self {
        Self {
            bloom_filters: Vec::new(),
            data,
            m,
        }
    }

    pub fn extend(&mut self) {
        let mut filter = BloomFilter::from_raw_parts(self.m, 1);
        self.data.iter().for_each(|d| filter.insert(d));

        self.bloom_filters.push(filter);
    }

    pub fn contains(&self, value: &T) -> bool {
        self.bloom_filters
            .iter()
            .all(|filter| filter.contains(value))
    }

    pub fn extend_until_stable_heuristic(
        &mut self,
        mut elements: Vec<T>,
        angle_threshold_deg: f64,
        window_size: usize,
        max_runs: usize,
    ) -> Result<(), Box<dyn Error>> {
        let mut recent_angles = VecDeque::with_capacity(window_size);
        let mut angle_sum = 0.0;
        let mut last_normalized: Option<f64> = None;

        for _ in 1..=max_runs {
            self.extend();

            let (positives, negatives): (Vec<T>, Vec<T>) =
                elements.into_iter().partition(|e| self.contains(e));

            let normalized = positives.len() as f64 / (positives.len() + negatives.len()) as f64;

            if let Some(prev) = last_normalized {
                let dy = normalized - prev;
                let angle = dy.abs().atan() * 180.0 / PI;

                if recent_angles.len() == window_size {
                    let removed = recent_angles.pop_front().unwrap();
                    angle_sum -= removed;
                }

                recent_angles.push_back(angle);
                angle_sum += angle;

                if recent_angles.len() == window_size {
                    let avg_angle = angle_sum / window_size as f64;
                    if avg_angle < angle_threshold_deg {
                        return Ok(());
                    }
                }
            }

            last_normalized = Some(normalized);
            elements = positives.into_iter().chain(negatives.into_iter()).collect(); // rebuild `elements` for next run
        }

        Err(Box::new(ConvergenceError(format!(
            "Did not converge within {max_runs} runs"
        ))))
    }

    pub fn extend_until_target_similarity(
        &mut self,
        receiver_data: Vec<T>,
        target_similarity: f64,
        max_runs: usize,
    ) -> Result<(), Box<dyn Error>> {
        if !(0.0..=1.0).contains(&target_similarity) {
            return Err(Box::new(ConvergenceError(format!(
                "Target similarity {target_similarity} is out of bounds (0.0 to 1.0)"
            ))));
        }

        let n_receiver = receiver_data.len();
        let mut receiver_bf = RatelessBF::new(receiver_data, self.m);

        let n_sender = self.data.len();

        for _ in 0..max_runs {
            self.extend();
            receiver_bf.extend();

            let mut inner_product = 0;
            for (a_filter, b_filter) in self
                .bloom_filters
                .iter()
                .zip(receiver_bf.bloom_filters.iter())
            {
                let mut tmp = a_filter.bitslice().to_bitvec();
                tmp &= b_filter.bitslice();
                inner_product += tmp.count_ones();
            }

            let estimated_intersection = estimate_intersection(
                inner_product as f64,
                n_receiver as i32,
                n_sender as i32,
                self.bloom_filters.len() as f64,
                self.m as f64,
            );

            let true_negatives = receiver_bf
                .data
                .iter()
                .filter(|e| !self.contains(e))
                .count();

            let similarity =
                (estimated_intersection as f64 + true_negatives as f64) / n_receiver as f64;
            eprintln!("Similarity {similarity} = {estimated_intersection} + {true_negatives}");

            if similarity >= target_similarity {
                return Ok(());
            }
        }

        Err(Box::new(ConvergenceError(format!(
            "Did not reach target similarity {target_similarity} in {max_runs} rounds"
        ))))
    }

    pub fn size_of(&self) -> usize {
        if self.bloom_filters.is_empty() {
            return 0;
        }

        let standalone_bf = &self.bloom_filters[0];
        let standalone_bf_size = standalone_bf.bitslice().chunks(8).count();

        self.bloom_filters.len() * standalone_bf_size //combined bitarray size in Bytes
        + mem::size_of::<u64>() //size to transmit m the number of bits (in each of the internal BFs)
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
