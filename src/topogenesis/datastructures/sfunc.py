
"""
Stencil Functions
"""

import numpy as np
import warnings
import os

__author__ = "Shervin Azadi, and Pirouz Nourian"
__copyright__ = "???"
__credits__ = ["Shervin Azadi", "Pirouz Nourian"]
__license__ = "???"
__version__ = "0.0.2"
__maintainer__ = "Shervin Azadi"
__email__ = "shervinazadi93@gmail.com"
__status__ = "Dev"

file_directory = os.path.dirname(os.path.abspath(__file__))

# directly using numpy functions
sum = np.sum

# wrapping numpy functions to enable mapping


def random_choice(array, axis=1):
    return np.apply_along_axis(np.random.choice, axis, array)
