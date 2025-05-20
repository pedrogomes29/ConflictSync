import numpy as np
from bitarray import bitarray
import string
from secrets import choice
import random

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

def count_true_negatives(elements, bf, m):
    return sum(bf[hash(el) % m] == False for el in elements)

# Parameters
len_a = 100_000
len_b = 5_000
subset_size = len_b
m = len_b
k = 1
rng = random.Random(42)

# Generate disjoint sets
a_elements, b_elements = generate_disjoint_sets(len_a, len_b, rng)
subset_a = rng.sample(a_elements, subset_size)

# Create Bloom filter for b
bf_b = generate_bloom_filter(b_elements, m)

# Estimate using full set
bf_a_full = generate_bloom_filter(a_elements, m)
observed_ip_full = (bf_a_full & bf_b).count(True)
estimated_intersection_full = estimate_intersection(observed_ip_full, len_b, len_a, k, m)
true_neg_full = count_true_negatives(a_elements, bf_b, m)
similarity_full = (true_neg_full + estimated_intersection_full) / len_a

# Estimate using subset
bf_a_subset = generate_bloom_filter(subset_a, m)
observed_ip_subset = (bf_a_subset & bf_b).count(True)
estimated_intersection_subset = estimate_intersection(observed_ip_subset, len_b, subset_size, k, m)
true_neg_subset = count_true_negatives(subset_a, bf_b, m)
similarity_subset = (true_neg_subset + estimated_intersection_subset) / subset_size

# Results
print(f"\n--- Results ---")
print(f"True Negatives (Full, via BF): {true_neg_full}")
print(f"Estimated Intersection (Full): {estimated_intersection_full}")
print(f"Similarity (Full): {similarity_full:.4f}\n")

print(f"True Negatives (Subset, via BF): {true_neg_subset}")
print(f"Estimated Intersection (Subset): {estimated_intersection_subset}")
print(f"Similarity (Subset): {similarity_subset:.4f}")

bits_set_in_bf_a_full = bf_a_full.count(True)
print(f"Bits set in bf_a_full: {bits_set_in_bf_a_full}")