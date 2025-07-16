import numpy as np

def obtain_posterior(xTx, xTy, prior_covariance, emission_variance_em):
    
    # Posterior covariance
    
    # ... obtain required precisions. n.b. efficient inversion of diagonal prior_covariance
    emission_precision = 1 / emission_variance_em
    prior_precision = np.diag(1 / np.diag(prior_covariance))
    
    # ... obtain posterior values
    posterior_precision = prior_precision + emission_precision * xTx
    posterior_covariance = np.linalg.inv(posterior_precision)
    
    # Posterior mean
    posterior_mean = emission_precision * np.dot(posterior_covariance, xTy)
    
    return posterior_mean, posterior_covariance

def manage_sparsity(X_em, mask_em, prior_covariance, posterior_mean, posterior_covariance):
    
    # ... dimensions before sparsity
    N, M = X_em.shape
    
    # ... arrays (reduced size)
    gamma = np.zeros(M)
    alpha = 1 / np.diag(prior_covariance)
    alpha_new = np.zeros(M)
    
    # ... running total
    sum_gamma = 0.0
    
    # ... loop through the basis functions that are still being used
    for m in range(M):
        
        # ... prior covariance values
        gamma[m] = 1.0 - alpha[m] * posterior_covariance[m, m]
        alpha_new[m] = gamma[m] / (posterior_mean[m] * posterior_mean[m])
        
        # ... test for sparsity
        big_number = 1e10
        
        if alpha_new[m] > big_number:
            
            alpha_new[m] = big_number
            gamma[m] = 0.0
            mask_em[m] = False
        
        # ... update the sum of gamma values
        sum_gamma += gamma[m]
        
    # ... remove basis functions
    X_em = X_em[:, mask_em]
    posterior_mean = posterior_mean[mask_em]
    alpha = alpha_new[mask_em]
    mask_em = mask_em[mask_em]
    
    return N, sum_gamma, alpha, mask_em, X_em, posterior_mean

def reestimate_parameter_values(y, X_em, sum_gamma, alpha, posterior_mean):
    
    # ... dimensions
    N, _ = X_em.shape
    
    # ... re-estimate the emission variance
    distance = y - np.dot(X_em, posterior_mean)
    distance_sq = np.dot(np.transpose(distance), distance)
    denominator = N - sum_gamma
    emission_variance = distance_sq / denominator
    
    # ... re-estimate the prior_covariance
    prior_covariance = np.diag(1 / alpha)
    
    return prior_covariance, emission_variance

def run_m_step(mask_em, y, X_em, prior_covariance, posterior_mean, posterior_covariance):
    
    # ... manage sparsity
    N, sum_gamma, alpha, mask_em, X_em, posterior_mean = manage_sparsity(X_em, mask_em, 
                                                                         prior_covariance, 
                                                                         posterior_mean, posterior_covariance)
    
    # ... re-estimate parameter values
    prior_covariance, emission_variance = reestimate_parameter_values(y, X_em, sum_gamma, alpha, posterior_mean)

    return prior_covariance, emission_variance, mask_em, X_em, posterior_mean
    

def run_em_algorithm(num_iterations, mask_em, y, X_em, prior_mean, prior_covariance, emission_variance_em):
    
    for i in range(num_iterations):
        
        # ... pre-calculate xTx, xTy. n.b. inside the loop because we remove columns from X_em for sparsity
        xTx = np.dot(np.transpose(X_em), X_em)
        xTy = np.dot(np.transpose(X_em), y)
        
        # ... E-step
        posterior_mean, posterior_covariance = obtain_posterior(xTx, xTy, prior_covariance, emission_variance_em)
        
        # ... M-step
        prior_covariance, emission_variance, mask_em, X_em, posterior_mean = run_m_step(mask_em, y, X_em, 
                                                          prior_covariance, posterior_mean, posterior_covariance)
    
    
    
    return posterior_mean, X_em