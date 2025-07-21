import hashlib
import matplotlib.pyplot as plt

# Parameters
m = 14  # Number of bits
k = 1   # Number of hash functions

def hash_k(element, seed=0):
    h = hashlib.sha256(f"{seed}-{element}".encode()).hexdigest()
    return int(h, 16) % m

def create_bloom_filter(elements):
    bit_array = [0] * m
    for el in elements:
        for i in range(k):
            index = hash_k(el, i)
            bit_array[index] = 1
    return bit_array

def draw_bloom_filters(filters, titles, main_title):
    num_filters = len(filters)
    fig, axes = plt.subplots(num_filters, 1, figsize=(7, 1.5 * num_filters))
    if num_filters == 1:
        axes = [axes]

    for ax, bits, title in zip(axes, filters, titles):
        for i, bit in enumerate(bits):
            color = 'black' if bit == 1 else 'white'
            edgecolor = 'gray'
            rect = plt.Rectangle((0.5*i, 0), 0.5, 0.5, facecolor=color, edgecolor=edgecolor)
            ax.add_patch(rect)
        ax.set_xlim(0, m)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(m))
        ax.set_yticks([])
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def bitwise_and(bf1, bf2):
    return [a & b for a, b in zip(bf1, bf2)]

# === Case 1: Same Elements ===
shared_elements = [f"item{i}" for i in range(10)]
bf1 = create_bloom_filter(shared_elements)
bf2 = create_bloom_filter(shared_elements)
and_same = bitwise_and(bf1, bf2)

draw_bloom_filters(
    [bf1, bf2, and_same],
    ["Bloom Filter 1 (Same Elements)", "Bloom Filter 2 (Same Elements)", "Bitwise AND"],
    "Case 1: Same Elements in Both Filters"
)

# === Case 2: Different Elements ===
elements1 = [f"C{i}" for i in range(10)]
elements2 = [f"D{i}" for i in range(10)]
bf3 = create_bloom_filter(elements1)
bf4 = create_bloom_filter(elements2)
and_diff = bitwise_and(bf3, bf4)

draw_bloom_filters(
    [bf3, bf4, and_diff],
    ["Bloom Filter 1 (A elements)", "Bloom Filter 2 (B elements)", "Bitwise AND"],
    "Case 2: Different Elements in Each Filter"
)