#!/usr/bin/env python
# encoding: utf-8
"""
Prototype script for MCMC -- just for proof-of-concept!

The mock dataset is an 5-parameter tau-model with e-folding time of 1,
with Z/Z_sol=0 (zmet=20). Only the e-folding time is unknown.
"""

import emcee
import numpy as np
import fsps

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.gridspec as gridspec

from sedbot.photconv import ab_to_mjy


def main():
    print "Initializing SP"
    n_walkers = 4
    n_dim = 1  # SSP age
    n_steps = 1000
    n_burn = 50  # burn-in of 50 steps

    pset_true = mock_parameters()
    bands = ['SDSS_u', 'SDSS_g', 'SDSS_r', 'SDSS_i']
    p0 = init_chain(n_walkers)
    d = 785 * 1000.  # distance in parsecs

    sp = fsps.StellarPopulation(**pset_true)
    mock_mags = sp.get_mags(bands=bands, tage=pset_true['tage'])
    mock_mjy = ab_to_mjy(mock_mags, d)
    sigma_mags = np.ones(len(bands)) * 0.1
    mock_sigma = (mock_mjy * sigma_mags) / 1.0875

    print "Mock data"
    for b, mjy, sigma in zip(bands, mock_mjy, mock_sigma):
        print "%s %.2e %.2e" % (b, mjy, sigma)

    print "Running emcee"
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, ln_prob,
            args=(mock_mjy, mock_sigma, sp, d, bands))
    sampler.run_mcmc(p0, n_steps)
    print "chain shape", sampler.flatchain.shape
    print "Mean acceptance fraction: {0:.3f}".format(
            np.mean(sampler.acceptance_fraction))

    fig = Figure(figsize=(3.5, 3.5))
    canvas = FigureCanvas(fig)
    gs = gridspec.GridSpec(2, 1, left=0.15, right=0.93, bottom=0.12, top=0.95,
        wspace=None, hspace=0.4, width_ratios=None, height_ratios=None)
    ax_age = fig.add_subplot(gs[0])
    ax_chain = fig.add_subplot(gs[1])

    steps = np.arange(n_steps)
    for i in xrange(n_walkers):
        chain = sampler.chain[i, :, 0]
        ax_chain.plot(steps, chain, '-', alpha=0.5)
    ax_chain.set_xlabel("steps")
    ax_chain.set_ylabel(r"$t$ (Gyr)")

    burnt_chain = sampler.chain[:, n_burn:, :].reshape((-1, n_dim))
    print "burnt_chain shape", burnt_chain.shape
    mean_age = burnt_chain.mean()
    std_age = burnt_chain.std()
    ax_age.hist(burnt_chain, bins=100, range=(0., 5),
        histtype='stepfilled', edgecolor='None', facecolor='0.5')
    ax_age.text(0.9, 0.9, "Mock Age %.2f Gyr" % pset_true['tage'],
        size=8,
        transform=ax_age.transAxes, ha='right', va='top')
    ax_age.text(0.9, 0.78, "Est. $%.2f \pm %.2f$ Gyr" % (
        mean_age, std_age),
        size=8,
        transform=ax_age.transAxes, ha='right', va='top')
    ax_age.axvline(x=pset_true['tage'], linewidth=1, color='r')
    ax_age.axvline(x=mean_age, linewidth=1, color='k', ls='--')
    ax_age.set_xlabel(r"$t$ (Gyr)")

    gs.tight_layout(fig, pad=1.08, h_pad=None, w_pad=None, rect=None)
    canvas.print_figure("proto_mcmc_chain.pdf", format="pdf")


def mock_parameters():
    """Generate a dict of FSPS settings to generate the mock data set."""
    return {'compute_vega_mags': False,
            'zmet': 20, # solar metallicity
            'sfh': 0,
            'tage': 1.,  # SSP age in Gyr
            }


def init_chain(n_walkers):
    """Init chain position given the number of walkers.
    
    These are initial guesses at SSP age (necessarily bad).
    """
    n_dim = 1
    p0 = np.random.randn(n_dim * n_walkers).reshape((n_walkers, n_dim)) + 5.
    p0[p0 < 0.1] = 0.1
    p0[p0 > 13.] = 13.
    return p0


def ln_prior(theta):
    """ln prior probability of age (theta). Uniform in age."""
    if theta > 0.1 and theta < 13.8:
        return 0.0
    else:
        return -np.inf


def ln_like(theta, obs_mjy, obs_sigma, sp, d, bands):
    """ln-likelihood function"""
    model_mags = sp.get_mags(tage=theta, bands=bands)
    model_mjy = ab_to_mjy(model_mags, d)
    L = -0.5 * np.sum(np.power((model_mjy - obs_mjy) / obs_sigma, 2.))
    return L


def ln_prob(theta, obs_mjy, obs_sigma, sp, d, bands):
    """ln-probability function"""
    lp = ln_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_like(theta, obs_mjy, obs_sigma, sp, d, bands)


if __name__ == '__main__':
    main()
