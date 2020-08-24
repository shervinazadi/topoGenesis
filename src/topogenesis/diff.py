
"""
Numerical Differentiation tools for volumetric data 
"""

import numpy as np

__author__ = "Shervin Azadi, and Pirouz Nourian"
__copyright__ = "???"
__credits__ = ["Shervin Azadi", "Pirouz Nourian"]
__license__ = "???"
__version__ = "0.0.2"
__maintainer__ = "Shervin Azadi"
__email__ = "shervinazadi93@gmail.com"
__status__ = "Dev"


def gradient(sfield):

    # partial derivatives
    dx = (sfield[:-2, 1:-1, 1:-1] - sfield[2:, 1:-1, 1:-1]) * 0.5
    dy = (sfield[1:-1, :-2, 1:-1] - sfield[1:-1, 2:, 1:-1]) * 0.5
    dz = (sfield[1:-1, 1:-1, :-2] - sfield[1:-1, 1:-1, 2:]) * 0.5

    # stack gradient
    g = np.stack([dx, dy, dz], axis=0)

    return(g)


def divergence(vfield):

    X = vfield[0]
    Y = vfield[0]
    Z = vfield[0]

    # partial derivatives
    dx = (X[:-2, 1:-1, 1:-1] - X[2:, 1:-1, 1:-1]) * 0.5
    dy = (Y[1:-1, :-2, 1:-1] - Y[1:-1, 2:, 1:-1]) * 0.5
    dz = (Z[1:-1, 1:-1, :-2] - Z[1:-1, 1:-1, 2:]) * 0.5

    # sum divergence
    d = dx + dy + dz

    return(d)
