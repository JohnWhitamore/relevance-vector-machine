import numpy as np

def generate_synthetic_weights(M, use_flag):
    
    # ... sample weights from a Gaussian distribution
    w = np.random.multivariate_normal(np.zeros(M), np.eye(M))
    
    # ... set weights to zero for basis functions being omitted
    w = np.multiply(w, use_flag)
    
    return w

def generate_synthetic_data(X, w, emission_variance_generative):
    
    # ... dimensions
    N, M = X.shape
    
    # ... random noise
    emission_st_dev_generative = np.sqrt(emission_variance_generative)
    emission_noise = np.random.normal(0, emission_st_dev_generative, size = N)
    
    # ... synthetic data
    synth_data = np.dot(X, w) + emission_noise
    
    return synth_data