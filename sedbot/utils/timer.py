#!/usr/bin/env python
# encoding: utf-8
"""


2015-03-10 - Created by Jonathan Sick
"""

import time

INTERVALS = [1, 60, 3600, 86400, 604800, 2419200, 29030400]
NAMES = [('second', 'seconds'),
         ('minute', 'minutes'),
         ('hour',   'hours'),
         ('day',    'days'),
         ('week',   'weeks'),
         ('month',  'months'),
         ('year',   'years')]


class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

    def __str__(self):
        # return humanize_time(self.interval, "seconds"))
        return "{0:.2e} sec".format(self.interval)


def humanize_time(amount, units):
    """
    Divide `amount` in time periods.
    Useful for making time intervals more human readable.

    Based on: https://github.com/liudmil-mitev/experiments/blob/master/time/humanize_time.py  # NOQA

    >>> humanize_time(173, "hours")
    [(1, 'week'), (5, 'hours')]

    >>> humanize_time(17313, "seconds")
    [(4, 'hours'), (48, 'minutes'), (33, 'seconds')]

    >>> humanize_time(90, "weeks")
    [(1, 'year'), (10, 'months'), (2, 'weeks')]

    >>> humanize_time(42, "months")
    [(3, 'years'), (6, 'months')]

    >>> humanize_time(500, "days")
    [(1, 'year'), (5, 'months'), (3, 'weeks'), (3, 'days')]
    """
    result = []

    unit = map(lambda a: a[1], NAMES).index(units)
    # Convert to seconds
    amount = amount * INTERVALS[unit]

    for i in range(len(NAMES)-1, -1, -1):
        a = amount // INTERVALS[i]
        if a > 0:
            result.append((a, NAMES[i][1 % a]))
            amount -= a * INTERVALS[i]

    return result
