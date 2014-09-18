#!/usr/bin/env python
# encoding: utf-8
"""
Gibbs sampling of a hierarchical model with multiple pixels having independent
star formation histories and a global scalar background that effects all
pixels.
"""


class MultiPixelGibbsBgModeller(object):
    """Runs a Gibbs sampler that alternates between sampling pixels and
    sampling a background bias that uniformly affects all pixels.
    """
    def __init__(self, seds, sed_errs, sed_lnpost):
        super(GibbsBgModeller, self).__init__(self)
    
    def sample(self, n_iters,
               theta0, bg0, d0,
               n_pixel_steps=10,
               n_cpu=None):
        # Initialize the posterior chains 
        for j in xrange(n_iters):
            # Step 1. Send jobs out to each pixel to sample the stellar pop
            # in each pixel given the last parameter estimates.
            # get the parameter chain for each pixel and the model flux

            # Step 2. Sample the background

            # Step 3. Sample a new distance.
            pass
