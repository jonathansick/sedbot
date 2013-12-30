#!/usr/bin/env python
# encoding: utf-8
"""
Library of built-in probability functions.
"""

import numpy as np
import scipy.stats


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

def ln_uniform_factory(lower, upper):
    """Log of uniform prior probability factory.
    
    Parameters
    ----------
    lower : float
        Lower bound of the uniform probability distribution.
    upper : float
        Upper bound of the uniform probability distribution.

    Returns
    -------
    func : function
        A function that accepts a random variable and returns the log of the
        uniform probability of that value.
        Returns `-numpy.inf` if the RV is outside bounds.
    """
    width = upper - lower
    def func(x):
        """Log of uniform prior probability."""
        if x >= lower and x <= upper:
            return scipy.stats.uniform.logpdf(x, loc=lower, scale=width)
        else:
            return -np.inf
    return func


def ln_normal_factory(mu, sigma, limits=None):
    """Log of normal prior probability factory.

    Parameters
    ----------
    lower : float
        Lower bound of the log-normal probability distribution.
    upper : float
        Upper bound of the log-normal probability distribution.
    limits : (2,) tuple (optional)
        Hard lower and upper boundaries on the random variable.

    Returns
    -------
    func : function
        A function that accepts a random variable and returns the log-normal
        probability of that value. If `limits` are set then an RV outside the
        bounds has a ln-probability `-numpy.inf`.
    """
    if limits:
        lower, upper = limits
        def f_limits(x):
            """Log of normal prior probability with hard limits."""
            if x >= lower and x <= upper:
                return scipy.stats.norm.logpdf(x, loc=mu, scale=sigma)
            else:
                return -np.inf
        return f_limits
    else:
        def f(x):
            """Log of normal prior probability."""
            return scipy.stats.norm.logpdf(x, loc=mu, scale=sigma)
        return f
