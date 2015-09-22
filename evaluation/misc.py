__author__ = 'korhammer'

import numpy as np


def based_floor(x, base=1):
    """
    rounds down to the last multiple of base 10 (or other)
    """
    return base * np.floor(np.float(x) / base)


def based_ceil(x, base=1):
    """
    rounds up to the next multiple of base 10 (or other)
    """
    return base * np.ceil(np.float(x) / base)


def float(x):
    return np.float(x) if len(x) > 0 else np.nan


def int(x):
    return np.int(x) if len(x) > 0 else 0

