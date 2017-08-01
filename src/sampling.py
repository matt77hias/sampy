import numpy as np

###############################################################################
# Sampling: single sample
###############################################################################
def uniform_sample_on_unit_circle(u1):
    phi = 2.0 * np.pi * u1
    return np.array([np.cos(phi), np.sin(phi)])
    
def uniform_sample_in_unit_circle(u1, u2):
    return np.sqrt(u2) * uniform_sample_on_unit_circle(u1)
    
def uniform_sample_on_unit_halfcircle(u1):
    phi = np.pi * u1
    return np.array([np.cos(phi), np.sin(phi)])

def uniform_sample_in_unit_halfcircle(u1, u2):
    return np.sqrt(u2) * uniform_sample_on_unit_halfcircle(u1) 

def uniform_sample_on_unit_sphere(u1, u2):
    cos_theta = 1.0 - 2.0 * u1
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta * cos_theta))
    phi = 2.0 * np.pi * u2
    return np.array([np.cos(phi) * sin_theta, np.sin(phi) * sin_theta, cos_theta], dtype=np.float64)

def uniform_sample_in_unit_sphere(u1, u2, u3):
    return pow(u3, 1.0/3.0) * uniform_sample_on_unit_sphere(u1, u2)

def uniform_sample_on_unit_hemisphere(u1, u2):
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - u1 * u1))
    phi = 2.0 * np.pi * u2
    return np.array([np.cos(phi) * sin_theta, np.sin(phi) * sin_theta, u1], dtype=np.float64)
    
def uniform_sample_in_unit_hemisphere(u1, u2, u3):
    return pow(u3, 1.0/3.0) * uniform_sample_on_unit_hemisphere(u1, u2) 
    	
def cosine_weighted_sample_on_unit_hemisphere(u1, u2):
    cos_theta = np.sqrt(1.0 - u1)
    sin_theta = np.sqrt(u1)
    phi = 2.0 * np.pi * u2
    return np.array([np.cos(phi) * sin_theta, np.sin(phi) * sin_theta, cos_theta], dtype=np.float64)

###############################################################################
# Sampling: multiple samples
###############################################################################
def uniform_samples_on_unit_circle(nb_samples, rng):
    ss = np.zeros((nb_samples,2))
    for i in range(nb_samples):
        ss[i] = uniform_sample_on_unit_circle(rng.uniform())
    return ss
    
def uniform_samples_in_unit_circle(nb_samples, rng):
    ss = np.zeros((nb_samples,2))
    for i in range(nb_samples):
        ss[i] = uniform_sample_in_unit_circle(rng.uniform(), rng.uniform())
    return ss
    
def uniform_samples_on_unit_halfcircle(nb_samples, rng):
    ss = np.zeros((nb_samples,2))
    for i in range(nb_samples):
        ss[i] = uniform_sample_on_unit_halfcircle(rng.uniform())
    return ss
    
def uniform_samples_in_unit_halfcircle(nb_samples, rng):
    ss = np.zeros((nb_samples,2))
    for i in range(nb_samples):
        ss[i] = uniform_sample_in_unit_halfcircle(rng.uniform(), rng.uniform())
    return ss
    
def uniform_samples_on_unit_sphere(nb_samples, rng):
    ss = np.zeros((nb_samples,3))
    for i in range(nb_samples):
        ss[i] = uniform_sample_on_unit_sphere(rng.uniform(), rng.uniform())
    return ss
    
def uniform_samples_in_unit_sphere(nb_samples, rng):
    ss = np.zeros((nb_samples,3))
    for i in range(nb_samples):
        ss[i] = uniform_sample_in_unit_sphere(rng.uniform(), rng.uniform(), rng.uniform())
    return ss
    
def uniform_samples_on_unit_hemisphere(nb_samples, rng):
    ss = np.zeros((nb_samples,3))
    for i in range(nb_samples):
        ss[i] = uniform_sample_on_unit_hemisphere(rng.uniform(), rng.uniform())
    return ss
    
def uniform_samples_in_unit_hemisphere(nb_samples, rng):
    ss = np.zeros((nb_samples,3))
    for i in range(nb_samples):
        ss[i] = uniform_sample_in_unit_hemisphere(rng.uniform(), rng.uniform(), rng.uniform())
    return ss
    
def cosine_weighted_samples_on_unit_hemisphere(nb_samples, rng):  
    ss = np.zeros((nb_samples,3))
    for i in range(nb_samples):
        ss[i] = cosine_weighted_sample_on_unit_hemisphere(rng.uniform(), rng.uniform())
    return ss