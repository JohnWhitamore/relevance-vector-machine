import numpy as np
import matplotlib.pyplot as plt

import xdata
import synthdata
import em

"""
This code reflects the Relevance Vector Machine of Mike Tipping.
The idea is to start with a "full" set of basis functions and then
to prune them in successful iterations of the EM algorithm.

Note that the pruning necessitates that care be taken over the
certain quantities, including the design matrix, X, whose
columns each represent a basis function.
"""

"""
Data and basis functions
"""

# Specify calendar parameter values

# ... calendar
num_years = 2
days_per_year = 365
seasons_per_year = 4

# ... number of time-steps
num_time_steps = days_per_year * num_years

# Generate basis functions
# ... note that X represents the "full" design matrix with all basis functions included
X = xdata.generate_basis_functions(num_time_steps, seasons_per_year, days_per_year, include_bias=True)

# ... obtain dimensions
N, M = X.shape

"""
Synthetic data generation
"""

# ... randomly create mask that specifies which basis functions to use
probability_of_omission = 0.25
mask_gen = np.random.binomial(n=1, p=1-probability_of_omission, size=M).astype(bool)
print(mask_gen)

# ... generate synthetic weights
# ... note that w_gen and X_gen reflect only the basis functions that are being used
w_gen, X_gen = synthdata.generate_synthetic_weights(X, M, mask_gen)


# ... generate synthetic observed data
emission_variance_generative = 0.0001
y = synthdata.generate_synthetic_data(X_gen, w_gen, emission_variance_generative)
mean_generative = np.dot(X_gen, w_gen)

"""
Inference and learning
"""

# Specify inference parameter values

# ... number of iterations of the EM algorithm
num_iterations = 10

# ... initialise the design matrix and basis function mask to use within the EM algorithm
mask_em = np.ones(M, dtype=bool)
X_em = X[:, mask_em].copy()
_, M_em = X_em.shape

# ... initial estimates of parameter values
# ... - can often bootstrap estimates from a simpler model
emission_variance_em = emission_variance_generative * 1.2
prior_mean = np.zeros(M_em)
prior_precisions = np.ones(M_em)

# ... obtain the prior covariance from the prior precision. n.b. diagonal matrices
prior_covariance = np.diag(1 / prior_precisions)

# Run the EM algorithm
posterior_mean, X_em = em.run_em_algorithm(num_iterations, mask_em, 
                                      y, X_em, 
                                      prior_mean, prior_covariance, emission_variance_em)

# Predictive density
predictive_mean = np.dot(X_em, posterior_mean)

# Display output
plt.plot(np.arange(N), y, 'go', alpha = 0.4)
plt.plot(np.arange(N), mean_generative, 'b-')
plt.plot(np.arange(N), predictive_mean, 'k-')
plt.show()

