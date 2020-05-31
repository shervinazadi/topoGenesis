

"""
An example of exporting numpy array holding volumetric information into csvs
"""

import networkx as nx
import numpy as np
import pandas as pd

__author__ = "Shervin Azadi"
__copyright__ = "???"
__credits__ = ["Shervin Azadi"]
__license__ = "???"
__version__ = "0.0.1"
__maintainer__ = "Shervin Azadi"
__email__ = "shervinazadi93@gmail.com"
__status__ = "Dev"

####################################################
# init an example of Boolean volumtric data
# containing information of a filled sphere
####################################################

# init the 3 dimensinal array
vs = 10
vol_shape = np.array([vs, vs, vs])
vol = np.zeros(vol_shape).astype(int)

# get the indicies of the voxels
vol_3d_ind = np.indices(vol.shape)
vol_3d_ind_flat = np.c_[vol_3d_ind[0].ravel(
), vol_3d_ind[1].ravel(), vol_3d_ind[2].ravel()]

# set the values to 1 if it is inside a sphere
rad = 5
vol_p2 = np.sum(np.power(vol_3d_ind_flat, 2), axis=1).reshape(vol_shape)
np.place(vol, vol_p2 <= rad ** 2, 1)

####################################################
# create panda dataframe
####################################################

vol_flat = vol.flatten()

vol_df = pd.DataFrame(
    {'IX': vol_3d_ind_flat[:, 0],
     'IY': vol_3d_ind_flat[:, 1],
     'IZ': vol_3d_ind_flat[:, 2],
     'value': vol_flat,
     })

####################################################
# save to CSV
####################################################

vol_df.to_csv('PY_OUT/volume.csv', index=False, float_format='%g')
