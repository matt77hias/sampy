from plot_utils import vis_samples_2D, vis_samples_3D
import numpy as np
import sampling as sp

###############################################################################
# Test
###############################################################################
def test():
    rng = np.random
    vis_samples_2D(sp.uniform_samples_on_unit_circle(nb_samples=128, rng=rng))
    vis_samples_2D(sp.uniform_samples_in_unit_circle(nb_samples=256, rng=rng)) 
    vis_samples_2D(sp.uniform_samples_on_unit_halfcircle(nb_samples=128, rng=rng))  
    vis_samples_2D(sp.uniform_samples_in_unit_halfcircle(nb_samples=256, rng=rng))  
    vis_samples_3D(sp.uniform_samples_on_unit_sphere(nb_samples=256, rng=rng))  
    vis_samples_3D(sp.uniform_samples_in_unit_sphere(nb_samples=256, rng=rng)) 
    vis_samples_3D(sp.uniform_samples_on_unit_hemisphere(nb_samples=256, rng=rng))  
    vis_samples_3D(sp.uniform_samples_in_unit_hemisphere(nb_samples=256, rng=rng))
    vis_samples_3D(sp.cosine_weighted_samples_on_unit_hemisphere(nb_samples=256, rng=rng))