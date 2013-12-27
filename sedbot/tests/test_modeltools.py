#!/usr/bin/env python
# encoding: utf-8
"""
Tests for the :mod:`sedbot.modeltools` module.
"""

import numpy as np
from sedbot.modeltools import reset_seed_limits


def test_reset_seed_limits():
    arr = np.zeros((3, 2), dtype=np.float)
    limits = (0., 10.)
    arr[0, 1] = -1.
    arr[1, 1] = 0.
    arr[2, 1] = 100.
    reset_seed_limits(arr[:, 1], *limits)
    assert arr[0, 0] == 0.
    assert arr[0, 1] == limits[0]
    assert arr[1, 1] == 0.
    assert arr[2, 1] == limits[1]
