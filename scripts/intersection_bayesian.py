import numpy as np
from scipy.stats import binom

def theta(x, n_a, n_b, k, m):
    t = lambda n: (1 - 1/m) ** (k * n)
    return 1 - t(n_a) - t(n_b) + t(n_a + n_b - x)

def posterior_p_x_given_y(y_obs, threshold_x, n_a, n_b, k, m, prior=None):
    n_min = min(n_a, n_b)
    xs = np.arange(0, n_min + 1)

    # Compute θ(x) and likelihoods
    thetas = np.array([theta(x, n_a, n_b, k, m) for x in xs])
    likelihoods = binom.pmf(y_obs, m, thetas)

    # Assume uniform prior if none given
    if prior is None:
        prior = np.ones_like(xs)
    else:
        prior = np.array(prior)

    unnormalized = likelihoods * prior
    posterior = unnormalized / unnormalized.sum()

    # Return cumulative probability for x >= threshold
    idx = np.searchsorted(xs, threshold_x)
    return posterior[idx:].sum()

# Example
n_a = 100_000
n_b = 100_000
k = 1
m = 100_000
y_obs = 45458       # Observed bits set in AND filter
threshold_x = 33000 # We want Pr[X >= 20000 | Y = 45458]


def estimate_intersection(observed_inner_product, n_receiver, n_sender, k, m):
    y = 1.0 - 1.0 / m
    numerator = (
        observed_inner_product / (k * m)
        + y ** n_receiver
        + y ** n_sender
        - 1.0
    )
    if numerator <= 0:
        return 0
    return n_receiver + n_sender - round(np.log(numerator) / np.log(y))

estimated = estimate_intersection(y_obs, n_a, n_b, k ,m)
p = posterior_p_x_given_y(y_obs, estimated, n_a, n_b, k, m)
print(f"P(X ≥ {estimated} | Y = {y_obs}) ≈ {p:.5f}")
