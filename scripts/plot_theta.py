import numpy as np
import matplotlib.pyplot as plt
from bitarray import bitarray
import string
from secrets import choice
import random
import math
from scipy.stats import beta as beta_dist

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

def compute_theta_min(n_sender, n_receiver, true_neg, m, target_similarity):
    desired_intersection =  math.ceil(target_similarity * n_receiver - true_neg)
    print(desired_intersection)
    y = 1.0 - 1.0 / m
    theta_min = 1.0 - y**n_sender - y**n_receiver + y**(n_sender + n_receiver - desired_intersection)
    return theta_min

def generate_bloom_filter(elements, m, slice_idx):
    bf = bitarray(m)
    bf.setall(False)
    for el in elements:
        h = hash(el + str(slice_idx)) % m
        bf[h] = True
    return bf

# --- Simulation Parameters ---
len_a = 30
len_b = 30
m = math.ceil(len_a*(1/math.log(2)))
target_similarity = 0.99
confidence = 0.75
rng = random.Random(42)

a_elements, b_elements = generate_disjoint_sets(len_a, len_b, rng)

# Bayesian prior
alpha, beta = 1, 1

# Tracking
slice_count = 0
alpha_vals = [alpha]
beta_vals = [beta]
theta_mins = [compute_theta_min(len_a, len_b, 0, m, target_similarity)]
negatives_per_slice = [0]

rateless_bf_a = []

# --- Synchronization loop ---
slice_idx = 0

while True:
    bf_a = generate_bloom_filter(a_elements, m, slice_idx)
    bf_b = generate_bloom_filter(b_elements, m, slice_idx)
    
    slice_idx += 1
    rateless_bf_a.append(bf_a)

    nr_negatives = sum(
        any(bf[hash(el + str(idx)) % m] == False for idx, bf in enumerate(rateless_bf_a))
        for el in b_elements
    )

    theta_min = compute_theta_min(len_a, len_b, nr_negatives, m, target_similarity)
    
    y = (bf_a & bf_b).count(True)
    
    alpha += y
    beta += (m - y)
    slice_count += 1

    alpha_vals.append(alpha)
    beta_vals.append(beta)
    theta_mins.append(theta_min)
    negatives_per_slice.append(nr_negatives)
    
    p = 1 - beta_dist.cdf(theta_min, alpha, beta)
    print(p)

    if p >= confidence:
        break


x = np.linspace(0, 1, 500)
n_plots = len(alpha_vals)

# Calculate the layout grid
n_cols = math.ceil(math.sqrt(n_plots))
n_rows = math.ceil(n_plots / n_cols)

# Precompute max y for consistent scale
max_y = 0
for a, b in zip(alpha_vals, beta_vals):
    y = beta_dist.pdf(x, a, b)
    max_y = max(max_y, np.max(y))

# Create plot with consistent y-axis limits
plt.figure(figsize=(4 * n_cols, 3 * n_rows))

for idx in range(n_plots):
    a = alpha_vals[idx]
    b = beta_vals[idx]
    th = theta_mins[idx]
    negs = negatives_per_slice[idx]
    y = beta_dist.pdf(x, a, b)
    
    plt.subplot(n_rows, n_cols, idx + 1)
    plt.plot(x, y, label=f'Beta({a}, {b})')
    
    y_val = beta_dist.pdf(th, a, b)
    plt.plot(th, y_val, 'ro', label=f'θ_min = {th:.4f}')
    
    plt.ylim(0, max_y)
    plt.title(f'Slice {idx} | Negs = {negs}', fontsize=8)
    plt.xlabel('θ')
    plt.ylabel('Density')
    plt.legend(fontsize=6)

plt.tight_layout()
plt.show()

