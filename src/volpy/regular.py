
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


def regularization(point_cloud, voxel_size, **kwargs):

    ####################################################
    # INPUTS
    ####################################################

    return_vol = kwargs.get('return_vol', False)

    ####################################################
    # METHOD
    ####################################################

    # finding the closest voxel to each point
    vox_ind = np.rint(point_cloud / voxel_size)
    # removing repetitions
    unique_vox_ind = np.unique(vox_ind, axis=0).astype(int)
    # mapping the voxel indicies to real space
    reg_pnt = unique_vox_ind * voxel_size

    if return_vol:

        # initializing the volume
        min_ind = np.min(unique_vox_ind, axis=0)
        max_ind = np.max(unique_vox_ind, axis=0)
        vol = np.zeros(1 + max_ind - min_ind).astype(int)
        # mapp the indicies to start from zero
        mapped_ind = unique_vox_ind - min_ind
        # setting the occupied voxels to 1
        vol[mapped_ind[:, 0], mapped_ind[:, 1], mapped_ind[:, 2]] = 1

        return (reg_pnt, vol)
    else:
        return(reg_pnt)
