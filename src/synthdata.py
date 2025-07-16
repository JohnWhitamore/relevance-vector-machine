import numpy as np

def generate_synthetic_weights(X, M, mask_gen):
    
    # ... sample weights from a Gaussian distribution
    w = np.random.multivariate_normal(np.zeros(M), np.eye(M))
    
    # ... apply the basis function mask so that we use only the selected basis functions
    w_gen = w[mask_gen]
    X_gen = X[:, mask_gen]
    
    return w_gen, X_gen

def generate_synthetic_data(X_gen, w_gen, emission_variance_generative):
    
    # ... dimensions of the reduced-dimension design matrix X_gen
    N, _ = X_gen.shape
    
    # ... random noise
    emission_st_dev_generative = np.sqrt(emission_variance_generative)
    emission_noise = np.random.normal(0, emission_st_dev_generative, size = N)
    
    # ... synthetic data
    synth_data = np.dot(X_gen, w_gen) + emission_noise
    
    return synth_data