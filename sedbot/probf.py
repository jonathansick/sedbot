#!/usr/bin/env python
# encoding: utf-8
"""
Library of built-in probability functions.

These probability functions use `scipy.stats` at their core, but also
encapsulate *limits* so that they can return a ln prob of -infinity when
a sample is called outside those limits. This is useful for emcee sampling.

Note this is a recent refactoring from the old style where used function
factories. Now we just create objects that when
called give the probability, and have a sample() method to take random samples
from the finite probability domain.
"""

import numpy as np
import scipy.stats

from sedbot.photconv import sb_to_mass


class RandomVariable(object):
    """Base class for random variables that encapsulate a `scipy.stats`
    random variable.

    All superclasses must provide

    - self._rv - the `scipy.stats` random variable instance
    - self._limits - (optional) a 2-tuple of lower and upper limits on values
      that the random variable can take.
    """
    def __init__(self):
        super(RandomVariable, self).__init__()
        self._limits = None

    def __call__(self, x):
        if not self._limits:
            return self._rv.logpdf(x)
        elif x >= self._limits[0] and x <= self._limits[1]:
            return self._rv.logpdf(x)
        else:
            return -np.inf

    def sample(self, shape=None):
        if not shape:
            return self._rv.rvs()
        else:
            return self._rv.rvs(shape)


class LnUniform(RandomVariable):
    r"""Log of uniform probability.

    .. math::
       \ln p(x|x_1, x_2) = \ln \frac{1}{x_2 - x_1}

    Parameters
    ----------
    lower : float
        Lower bound of the uniform probability distribution.
    upper : float
        Upper bound of the uniform probability distribution.
    """
    def __init__(self, lower, upper):
        super(LnUniform, self).__init__()
        self._limits = (lower, upper)
        self._rv = scipy.stats.uniform(loc=lower, scale=upper - lower)


class LnUniformMass(LnUniform):
    """Log of uniform probability intended to be used as an uninformative
    prior on the log-mass given a range of log M/L.

    Parameters
    ----------
    logml_min : float
        Minimum log M/L value.
    logml_max : float
        Maximum log M/L value.
    sb : float
        Surface brightness, mag / arcsec^2.
    D_pc : float
        Distance in parsecs.
    area : float
        Area of pixel, in square arcsecs.
    msun : float
        Solar magnitude. With python-fsps this can be obtained using
        ``fsps.get_filter(band_name).msun_ab``.
    """
    def __init__(self, logml_min, logml_max, sb, D_pc, area, msun):
        low_mass = sb_to_mass(sb, msun, logml_min, area, D_pc)
        high_mass = sb_to_mass(sb, msun, logml_max, area, D_pc)
        super(LnUniformMass, self).__init__(low_mass, high_mass)


class LnNormal(RandomVariable):
    r"""Log of normal prior probability factory.

    .. math::
       \ln p(x|\mu, \sigma) = \ln \frac{1}{\sqrt{2 \pi \sigma^2}}
       e^{- \left( \frac{x - \mu}{2 \pi \sigma^2} \right)}

    Parameters
    ----------
    mu : float
        Mean
    sigma : float
        Standard deviation of Gaussian.
    limits : (2,) tuple (optional)
        Hard lower and upper boundaries on the random variable.
    """
    def __init__(self, mu, sigma, limits=None):
        super(LnNormal, self).__init__()
        self._limits = limits
        self._rv = scipy.stats.norm(loc=mu, scale=sigma)


def ln_uniform_factory(lower, upper):
    """Log of uniform prior probability factory (deprecated)."""
    return LnUniform(lower, upper)


def ln_normal_factory(mu, sigma, limits=None):
    """Log of normal prior probability factory (deprecated)."""
    return LnNormal(mu, sigma, limits=limits)


def ln_loguniform_factory(lower, upper):
    r"""Log of log-uniform prior probability factory.

    .. math::
       \ln p(x|x_1, x_2) = \ln \frac{1}{x \ln \left( x_1 / x_2 \right)}

    Parameters
    ----------
    lower : float
        Lower bound of the log-uniform probability distribution.
    upper : float
        Upper bound of the log-uniform probability distribution.

    Returns
    -------
    func : function
        A function that accepts a random variable and returns the log of the
        log-uniform probability of that value.
        Returns `-numpy.inf` if the RV is outside bounds.
    """
    factor = 1. / np.log(upper / lower)
    assert np.isfinite(factor), "log-uniform prior not finite"

    def func(x):
        """Log of uniform prior probability."""
        if x >= lower and x <= upper:
            return np.log(factor / x)
        else:
            return -np.inf
    return func
