
#Code to obtain minimum

from scipy.optimize import minimize_scalar
import numpy as np

def f(x):
    return (1 - np.exp(-1/x))**(1/x)

result = minimize_scalar(f, bounds=(0.01, 5), method='bounded')
print("Minimum at x =", result.x)
print("Minimum value =", result.fun)
"""

import numpy as np
import matplotlib.pyplot as plt

# Extended m_ratio range from just above 0 to 1000
mratio_values = np.linspace(0.01, 5, 1000)

# Compute the approximate false positive rate
fpr_approx = (1 - np.exp(-1 / mratio_values)) ** (1 / mratio_values)

# Optimal m_ratio: 1 / ln(2)
optimal_mratio = 1 / np.log(2)
optimal_fpr = (1 - np.exp(-1 / optimal_mratio)) ** (1 / optimal_mratio)

# Plot the approximation
plt.figure(figsize=(10, 6))
plt.plot(mratio_values, fpr_approx, label=r'$\left(1 - e^{-1 / m_{\mathrm{ratio}}} \right)^{1 / m_{\mathrm{ratio}}}$', color='darkblue')
plt.plot(optimal_mratio, optimal_fpr, 'ro', label=r'Optimal $m_{\mathrm{ratio}} = \frac{1}{\ln 2}$')

plt.xlabel(r'$m_{\mathrm{ratio}}$')
plt.ylabel('False Positive Probability (Approx.)')
plt.title('Approximate False Positive Rate vs $m_{\\mathrm{ratio}}$')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
"""