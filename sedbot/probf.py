#!/usr/bin/env python
# encoding: utf-8
"""
Library of built-in probability functions.
"""

import numpy as np
import scipy.stats


def ln_uniform_factory(lower, upper):
    """Log of uniform prior probability factory.
    
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
        uniform probability of that value. That is, an RV within the bounds has
        a ln-probability of 0, and -infinity if outside the bounds.
    """
    width = upper - lower
    def func(x):
        """Log of uniform prior probability."""
        if x >= lower and x <= upper:
            return scipy.stats.uniform.logpdf(x, loc=lower, scale=width)
        else:
            return -np.inf
    return func


def ln_normal_factory(mu, sigma):
    """Log of normal prior probability factory.

    Parameters
    ----------
    lower : float
        Lower bound of the log-normal probability distribution.
    upper : float
        Upper bound of the log-normal probability distribution.

    Returns
    -------
    func : function
        A function that accepts a random variable and returns the log-normal
        probability of that value. That is, an RV within the bounds has a 
        ln-probability of 0, and -infinity if outside the bounds.
    """
    def func(x):
        """Log of normal prior probability factory."""
        return scipy.stats.norm.logpdf(x, loc=mu, scale=sigma)
    return func
