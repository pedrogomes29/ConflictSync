import numpy as np
import matplotlib.pyplot as plt
from bitarray import bitarray
import string
from secrets import choice
import random
from scipy.stats import beta

# --- Utility Functions ---
def sample_string(rng, min_len=5, max_len=80):
    length = rng.randint(min_len, max_len)
    return ''.join(choice(string.ascii_letters + string.digits) for _ in range(length))

def generate_disjoint_sets(len_a, len_b, rng):
    seen = set()
    a = set()
    while len(a) < len_a:
        s = sample_string(rng)
        if s not in seen:
            seen.add(s)
            a.add(s)
    b = set()
    while len(b) < len_b:
        s = sample_string(rng)
        if s not in seen:
            seen.add(s)
            b.add(s)
    return list(a), list(b)

def generate_bloom_filter(elements, m):
    bf = bitarray(m)
    bf.setall(False)
    for el in elements:
        h = hash(el) % m
        bf[h] = True
    return bf

# --- Simulation Parameters ---
len_a = 100_000
len_b = 100_000
slice_size = 5000  # Sender transmits this many elements per slice
m = 5000  # Bloom filter size
theta_min = 0.8  # We want to stop when θ >= this with high probability
delta = 0.01     # Confidence level: stop when P(θ ≥ θ_min) > 1 - δ
rng = random.Random(42)

# Generate sets and initial receiver Bloom filter
a_elements, b_elements = generate_disjoint_sets(len_a, len_b, rng)
bf_b = generate_bloom_filter(b_elements, m)

# Bayesian prior: Beta(1, 1)
alpha, beta_param = 1, 1

# Track posterior updates
slice_count = 0
posterior_probs = []

# Loop until convergence criterion is met
while True:
    # Sample slice from a_elements
    subset_a = rng.sample(a_elements, slice_size)
    bf_a = generate_bloom_filter(subset_a, m)
    
    # Compute number of 1s in AND of the Bloom filters
    y = (bf_a & bf_b).count(True)
    
    # Update Beta posterior
    alpha += y
    beta_param += (m - y)
    slice_count += 1
    
    # Compute probability that θ ≥ θ_min under current posterior
    p_theta = 1 - beta.cdf(theta_min, alpha, beta_param)
    posterior_probs.append(p_theta)
    
    print(f"Slice {slice_count}: y = {y}, P(θ ≥ {theta_min}) = {p_theta:.4f}")
    
    if p_theta > 1 - delta:
        print(f"\nStopping criterion met after {slice_count} slices.")
        break

# --- Optional Plot ---
x = np.linspace(0, 1, 500)
pdf = beta.pdf(x, alpha, beta_param)
plt.plot(x, pdf, label=f'Beta({alpha}, {beta_param})')
plt.axvline(theta_min, color='red', linestyle='--', label=f'θ_min = {theta_min}')
plt.title("Posterior Distribution of θ")
plt.xlabel("θ (shared-bit probability)")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()
