import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zipf # Import the Zipf distribution from scipy.stats

def zipf_pmf(k, s, N):
    """
    Calculates the Probability Mass Function (PMF) for a Zipf distribution.

    Args:
        k (int or array-like): The rank(s) for which to calculate the probability.
                               Typically ranges from 1 to N.
        s (float): The exponent parameter of the Zipf distribution.
                   When s is very close to 0, it approximates a uniform distribution.
                   Higher s values lead to a steeper, more "Zipfian" distribution.
        N (int): The total number of elements or maximum rank in the distribution.
                 This N is used for proper normalization of the finite Zipf distribution.

    Returns:
        float or array-like: The probability P(X=k) for the given rank(s).
    """
    # scipy.stats.zipf.pmf is defined for s > 0.
    # When s is very close to 0, scipy.stats.zipf.pmf might still return values
    # but the exact finite-N normalization for a truly uniform distribution
    # is best handled by explicit calculation.
    # However, for plotting the *shape* and demonstrating the approximation,
    # we can use scipy.stats.zipf.pmf and then apply a custom normalization
    # to ensure probabilities sum to 1 over the specified N ranks,
    # especially for very small 's'.

    # Calculate unnormalized probabilities (proportional to 1/k^s) using scipy's pmf.
    # We multiply by zeta(s) to get the unnormalized 1/k^s terms if we were to
    # use the raw pmf output, but it's simpler to just get the relative values.
    # The 'a' parameter in scipy.stats.zipf.pmf corresponds to 's' here.
    unnormalized_probs = 1.0 / (np.array(k)**s) # This is the 1/k^s part
    
    # Calculate the normalization constant for a finite Zipf distribution
    # This is the Nth generalized harmonic number of order s.
    normalization_constant = np.sum(1.0 / np.arange(1, N + 1)**s)

    # Normalize the probabilities so they sum to 1 over the N elements
    return unnormalized_probs / normalization_constant


def main():
    """
    Main function to visualize Zipf distributions with varying 's' parameters.
    """
    N = 50  # Total number of elements/ranks for visualization range
    ranks = np.arange(1, N + 1) # Ranks from 1 to N

    # Different 's' values to demonstrate the transition
    # Using a very small positive 's' instead of exactly 0.0
    s_values = [1e-9, 0.5, 1.0, 1.5, 2.0] # 1e-9 is very close to 0

    plt.figure(figsize=(12, 8))

    for s in s_values:
        # Calculate probabilities for the current 's' value
        probabilities = zipf_pmf(ranks, s, N)

        # Plot the distribution
        plt.plot(ranks, probabilities, marker='o', linestyle='-', label=f's = {s:.1e}' if s < 0.1 else f's = {s:.1f}')
        # Using a marker helps to emphasize the discrete nature of the ranks.

    plt.title('Zipf Distribution with Varying Exponent (s)', fontsize=16)
    plt.xlabel('Rank (k)', fontsize=12)
    plt.ylabel('Probability P(X=k)', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Exponent (s)', fontsize=10, title_fontsize=12)
    plt.tight_layout()
    plt.show()

    # Optional: Log-log plot for s=1.0 (classic Zipf's Law) and a higher 's'
    plt.figure(figsize=(12, 8))
    s_log_values = [1.0, 2.0]
    for s in s_log_values:
        probabilities = zipf_pmf(ranks, s, N)
        plt.loglog(ranks, probabilities, marker='o', linestyle='-', label=f's = {s:.1f}')

    plt.title('Zipf Distribution on Log-Log Scale', fontsize=16)
    plt.xlabel('Log(Rank)', fontsize=12)
    plt.ylabel('Log(Probability)', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Exponent (s)', fontsize=10, title_fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()