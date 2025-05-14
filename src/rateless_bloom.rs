use std::{collections::VecDeque, error::Error, f64::consts::PI, fmt::{self, Display, Formatter}, hash::Hash, mem};
use super::bloom::BloomFilter;

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
    m_ratio: f64,
}

impl<T> RatelessBF<T>
where
    T: Hash,
{
    #[inline]
    #[must_use]
    pub fn new(data: Vec<T>, m_ratio: f64) -> Self {
        Self {
            bloom_filters: Vec::new(),
            data,
            m_ratio
        }
    }

    pub fn extend(&mut self) {
        let mut filter = BloomFilter::from_raw_parts((self.data.capacity() as f64 * self.m_ratio).ceil() as usize, 1);
        self.data.iter().for_each(|d| filter.insert(d));

        self.bloom_filters.push(filter);
    }

    pub fn contains(&self, value: &T) -> bool {
        self.bloom_filters
            .iter()
            .all(|filter| filter.contains(value))
    }

    pub fn extend_until_stable(
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
    
    pub fn size_of(&self) -> usize {
        if self.bloom_filters.is_empty(){
            return 0;
        }

        let standalone_bf = &self.bloom_filters[0];
        let standalone_bf_size = standalone_bf.bitslice().chunks(8).count();
        
        self.bloom_filters.len() * standalone_bf_size //combined bitarray size in Bytes
        + mem::size_of::<u64>() //size to transmit m the number of bits (in each of the internal BFs)
    }


}
