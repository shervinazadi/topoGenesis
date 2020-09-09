
"""
Stencil Functions
"""

import numpy as np
import warnings
import os

file_directory = os.path.dirname(os.path.abspath(__file__))

# directly using numpy functions
sum = np.sum
min = np.amin
argmin = np.argmin

# wrapping numpy functions to enable mapping


def random_choice(array, axis=1):
    return np.apply_along_axis(np.random.choice, axis, array)
