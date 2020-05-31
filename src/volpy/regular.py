
"""
Regularization of PointClouds
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


def regularization(point_cloud, voxel_size):
    # finding the closest voxel to each point
    vox_ind = np.rint(point_cloud / voxel_size)
    # removing repetitions
    unique_vox_ind = np.unique(vox_ind, axis=0)
    # mapping the voxel indicies to real space
    reg_pnt = unique_vox_ind * voxel_size

    return(reg_pnt)
