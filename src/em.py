import numpy as np

def obtain_posterior(xTx, xTy, prior_covariance, emission_variance):
    
    # Posterior covariance
    
    # ... obtain required precisions. n.b. efficient inversion of diagonal prior_covariance
    emission_precision = 1 / emission_variance
    prior_precision = np.diag(1 / np.diag(prior_covariance))
    
    # ... obtain posterior values
    posterior_precision = prior_precision + emission_precision * xTx
    posterior_covariance = np.linalg.inv(posterior_precision)
    
    # Posterior mean
    posterior_mean = emission_precision * np.dot(posterior_covariance, xTy)
    
    return posterior_mean, posterior_covariance

def run_m_step(y, X, prior_covariance, posterior_mean, posterior_covariance):
    
    # ... dimensions
    N, M = X.shape
    
    # ... arrays
    gamma = np.zeros(M)
    alpha = 1 / np.diag(prior_covariance)
    alpha_new = np.zeros(M)
    
    # ... running total
    sum_gamma = 0.0
    
    # ... loop through basis functions
    for m in range(M):
        
        # ... prior covariance values
        gamma[m] = 1.0 - alpha[m] * posterior_covariance[m, m]
        alpha_new[m] = gamma[m] / (posterior_mean[m] * posterior_mean[m])
        
        # ... update the sum of gamma values
        sum_gamma += gamma[m]
        
    # ... emission variance values
    distance = y - np.dot(X, posterior_mean)
    distance_sq = np.dot(np.transpose(distance), distance)
    denominator = N - sum_gamma
    emission_precision = distance_sq / denominator
    
    # ... re-estimated parameter values
    prior_covariance = np.diag(1 / alpha_new)
    emission_variance = 1 / emission_precision
    
    return prior_covariance, emission_variance
    

def run_em_algorithm(num_iterations, y, X, prior_mean, prior_covariance, emission_variance):
    
    # ... pre-calculate xTx, xTy
    xTx = np.dot(np.transpose(X), X)
    xTy = np.dot(np.transpose(X), y)
    
    for i in range(num_iterations):
        
        # ... E-step
        posterior_mean, posterior_covariance = obtain_posterior(xTx, xTy, prior_covariance, emission_variance)
        
        # ... M-step
        prior_covariance, emission_variance = run_m_step(y, X, prior_covariance, posterior_mean, posterior_covariance)
    
    return posterior_mean