import numpy as np

def set_season_parameter_values(seasons_per_year, days_per_year):
    
    # Allocate a central date and a lengthscale to each season
    
    # ... initialise
    season_centre = np.zeros(seasons_per_year)
    season_lengthscale = days_per_year // (2 * seasons_per_year)
    
    # ... loop through seasons
    for s in range(seasons_per_year):
        
        season_centre[s] = s * days_per_year // seasons_per_year
    
    return season_centre, season_lengthscale

def evaluate_radial_basis_function(n, centre, lengthscale, days_per_year):
    
    # Gaussian RBF taking into account that the year is a circle
    
    # ... circular year
    day_of_this_year = n % days_per_year
    day_of_prev_year = day_of_this_year + days_per_year
    day_of_next_year = day_of_this_year - days_per_year
    
    days_to_centre = min(abs(day_of_prev_year - centre), 
                         abs(day_of_this_year - centre),
                         abs(day_of_next_year - centre))
    
    # ... Gaussian radial basis function
    quadratic = -1.0 * days_to_centre * days_to_centre / (2.0 * lengthscale * lengthscale)
    normalisation = lengthscale * np.sqrt(2.0 * np.pi)
    rbf_value = np.exp(quadratic) / normalisation
    
    return rbf_value

def generate_basis_functions(num_time_steps, seasons_per_year, days_per_year, include_bias=True):
    
    # Set dimensions
    N = num_time_steps
    M = seasons_per_year
    
    if include_bias:
        M = seasons_per_year + 1
        
    # Instantiate array
    X = np.zeros([N, M])
    
    # Set season parameter values
    season_centre, season_lengthscale = set_season_parameter_values(seasons_per_year, days_per_year)
    
    # Loop through time-steps
    for n in range(N):
        
        # ... seasons
        for s in range(seasons_per_year):
            
            X[n, s] = evaluate_radial_basis_function(n, season_centre[s], season_lengthscale, days_per_year)
        
        # ... bias
        if include_bias:
            X[n, M-1] = 1.0
    
    
    return X
    
    