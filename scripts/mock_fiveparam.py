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
from sedbot.modeltools import EmceeTimer, burnin_flatchain, mock_dataset


def main():
    print "Initializing SP"
    n_walkers = 16 * fiveparam.NDIM
    n_steps = 100
    n_burn = 50  # burn-in of 50 steps

    d0 = 785 * 1000.  # distance in parsecs
    d0_sigma = 25. * 1000.
    m0 = 1. # total stellar mass
    logZZsol0 = -0.1
    pset_true = {'compute_vega_mags': False,
            'tau': 5., 'const': 0.1, 'sf_start': 0.5,
            'tburst': 11., 'fburst': 0.05, 'dust2': 0.2}
    limits = {'tau': (0.1, 100.), 'const': (0., 0.2), 'sf_start': (0.1, 3.),
            'tburst': (6, 13.), 'fburst': (0., 0.1), 'logZZsol': (-1.98, 0.2),
            'mass': (0.1, 2.),
            'd': (d0 - 3. * d0_sigma, d0 + 3. * d0_sigma),
            'dust2': (0., 1.)}
    param_names = ['mass', 'logZZsol', 'd', 'tau', 'const', 'sf_start',
            'tburst', 'fburst', 'dust2']

    bands = ['SDSS_u', 'SDSS_g', 'SDSS_r', 'SDSS_i']
    p0 = fiveparam.init_chain(n_walkers, d0, m0, limits)
    sp = fsps.StellarPopulation(**pset_true)
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

    # Generate Mock dataset
    mock_mjy, mock_sigma = mock_dataset(sp, bands, d0, m0, logZZsol0,
            0.05, apply_errors=True)

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

    chain_plot("chain", sampler,
        [escape_latex(n) for n in param_names],
        [limits[n] for n in param_names])
    triangle_plot("triangle",
        burnin_flatchain(sampler, n_burn),
        [escape_latex(n) for n in param_names],
        [limits[n] for n in param_names],
        figsize=(5, 5),
        truths=(m0, logZZsol0, d0, pset_true['tau'], pset_true['const'],
            pset_true['sf_start'], pset_true['tburst'], pset_true['fburst'],
            pset_true['dust2']))


if __name__ == '__main__':
    main()
