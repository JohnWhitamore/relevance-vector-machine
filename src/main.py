import numpy as np
import matplotlib.pyplot as plt

import xdata
import synthdata
import em

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
X = xdata.generate_basis_functions(num_time_steps, seasons_per_year, days_per_year, include_bias=True)

# ... obtain dimensions
N, M = X.shape

"""
Synthetic data generation
"""

# ... specify how noisy the data should be
emission_variance_generative = 0.0001

# ... choose a basis function *not* to use
omit_basis_function = 2
use_flag = np.ones(M, dtype = bool)
# use_flag[omit_basis_function] = 0

# ... generate synthetic data
w = synthdata.generate_synthetic_weights(M, use_flag)
y = synthdata.generate_synthetic_data(X, w, emission_variance_generative)
mean_generative = np.dot(X, w)

"""
Inference and learning
"""

# Specify inference parameter values

# ... number of iterations of the EM algorithm
num_iterations = 1

# ... initial estimates of parameter values
# ... - can often bootstrap estimates from a simpler model
emission_variance = emission_variance_generative * 1.2
prior_mean = np.zeros(M)
prior_precisions = np.ones(M)
prior_variances = 1 / prior_precisions
prior_covariance = np.diag(prior_variances)

posterior_mean = em.run_em_algorithm(num_iterations, y, X, prior_mean, prior_covariance, emission_variance)
fitted_mean = np.dot(X, posterior_mean)







plt.plot(np.arange(N), y, 'go', alpha = 0.4)
plt.plot(np.arange(N), mean_generative, 'b-')
plt.plot(np.arange(N), fitted_mean, 'k-')
plt.show()

