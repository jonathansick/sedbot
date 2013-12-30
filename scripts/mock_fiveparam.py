#!/usr/bin/env python
# encoding: utf-8
"""
Mock test of the 5-parameter SFH model (sedbot.models.fiveparam) with SDSS
bands.
"""

import emcee
import numpy as np
import fsps

from sedbot.probf import ln_uniform_factory, ln_normal_factory
import sedbot.models.fiveparam as fiveparam
from sedbot.plots import chain_plot, triangle_plot, escape_latex
from sedbot.modeltools import EmceeTimer, burnin_flatchain, mock_dataset, \
        init_chain


def main():
    print "Initializing SP"
    # Setup chain parameters
    n_walkers = 32 * fiveparam.NDIM
    n_steps = 100
    n_burn = 50  # burn-in of 30 steps

    # Define the mock model
    bands = ['SDSS_u', 'SDSS_g', 'SDSS_r', 'SDSS_i']
    d0 = 785 * 1000.  # distance in parsecs
    d0_sigma = 25. * 1000.
    m0 = 1. # total stellar mass
    logZZsol0 = -0.1
    pset_true = {'compute_vega_mags': False,  # use AB mags
            # 'add_stellar_remnants': 0,  # for stellar mass calculation
            'tau': 5., 'const': 0.1, 'sf_start': 0.5,
            'tburst': 11., 'fburst': 0.05, 'dust2': 0.2}
    # Initialize FSPS
    sp = fsps.StellarPopulation(**pset_true)
    # Generate the mock SED
    mock_mjy, mock_sigma = mock_dataset(sp, bands, d0, m0, logZZsol0,
            0.05, apply_errors=True)

    # Setup emcee
    # Order of param_names matches that in the MCMC chain
    param_names = ['mass', 'logZZsol', 'd', 'tau', 'const', 'sf_start',
            'tburst', 'fburst', 'dust2']
    # limits defines hard parameter limits (and used for priors)
    limits = {'tau': (0.1, 20.), 'const': (0., 0.2), 'sf_start': (0.1, 3.),
            'tburst': (6, 13.), 'fburst': (0., 0.1), 'logZZsol': (-1.98, 0.2),
            'mass': (0.1, 2.),
            'd': (d0 - 3. * d0_sigma, d0 + 3. * d0_sigma),
            'dust2': (0., 1.)}
    # Initialize the chain starting point
    chain0 = [m0, -0.3, d0, 10., 0.2, 2., 6., 0.2, 0.2]
    sigma0 = [1., 0.2, 5. * 1000., 1., 0.1, 1., 1., 0.1, 0.2]
    lims = [limits[n] for n in param_names]
    p0 = init_chain(n_walkers, chain0, sigma0, lims)

    # Define priors
    prior_funcs = [
            ln_uniform_factory(*limits['mass']),
            ln_uniform_factory(*limits['logZZsol']),
            ln_normal_factory(d0, d0_sigma, limits=limits['d']),
            ln_uniform_factory(*limits['tau']),
            ln_uniform_factory(*limits['const']),
            ln_uniform_factory(*limits['sf_start']),
            ln_uniform_factory(*limits['tburst']),
            ln_uniform_factory(*limits['fburst']),
            ln_uniform_factory(*limits['dust2'])]

    print "Running emcee"
    sampler = emcee.EnsembleSampler(n_walkers,
            fiveparam.NDIM, fiveparam.ln_prob,
            args=(mock_mjy, mock_sigma, bands, sp, prior_funcs))
    with EmceeTimer(n_steps, n_walkers) as emceet:
        sampler.run_mcmc(p0, n_steps)
    print emceet
    print "chain shape", sampler.flatchain.shape
    print "Mean acceptance fraction: {0:.3f}".format(
            np.mean(sampler.acceptance_fraction))
    print "Acor result:"
    for name, ac in zip(param_names, sampler.acor):
        print "\t%s %.1f" % (name, ac)

    flatchain = burnin_flatchain(sampler, n_burn, append_mstar=True,
            append_mdust=True)

    chain_plot("chain", sampler,
        [escape_latex(n) for n in param_names],
        [limits[n] for n in param_names])
    triangle_plot("triangle",
        flatchain,
        [escape_latex(n) for n in param_names] + [r"$M_*$", r"$M_d$"],
        [limits[n] for n in param_names] + [(0.5, 1.5), (0.0, 0.5)],
        figsize=(5, 5),
        truths=(m0, logZZsol0, d0, pset_true['tau'], pset_true['const'],
            pset_true['sf_start'], pset_true['tburst'], pset_true['fburst'],
            pset_true['dust2'], None, None))


if __name__ == '__main__':
    main()
