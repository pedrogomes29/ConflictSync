use statrs::distribution::{Beta, ContinuousCDF};

pub fn probability_converged_beta_tail(
    alpha: f64,
    beta: f64,
    desired_intersection: i32,
    n_receiver: i32,
    n_sender: i32,
    m: i32,
) -> f64 {
    let y = 1.0 - 1.0 / m as f64;

    let theta_target = 1.0
        - y.powi(n_sender)
        - y.powi(n_receiver)
        + y.powi(n_sender + n_receiver - desired_intersection);

    if let Ok(beta_dist) = Beta::new(alpha, beta) {
        1.0 - beta_dist.cdf(theta_target)
    } else {
        0.0
    }
}

pub fn numeric_posterior_tail(
    observed_and_ones: usize,
    total_bits: usize,
    n_sender: usize,
    n_receiver: usize,
    m: usize,
    imin: usize,
    imax: usize,
) -> f64 {
    let y = 1.0 - 1.0 / (m as f64);

    fn log_binomial_pmf(k: usize, n: usize, p: f64) -> f64 {
        use statrs::function::gamma::ln_gamma;
        let ln_comb =
            ln_gamma((n + 1) as f64) - ln_gamma((k + 1) as f64) - ln_gamma((n - k + 1) as f64);
        ln_comb + (k as f64) * p.ln() + ((n - k) as f64) * (1.0 - p).ln()
    }

    let prior = 1.0 / ((imax - 0 + 1) as f64);
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..=imax {
        let theta = 1.0
            - y.powi(n_sender as i32)
            - y.powi(n_receiver as i32)
            + y.powi((n_sender + n_receiver - i) as i32);

        let pmf = (log_binomial_pmf(observed_and_ones, total_bits, theta)).exp();
        let weighted = pmf * prior;

        denominator += weighted;
        if i >= imin {
            numerator += weighted;
        }
    }

    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}
