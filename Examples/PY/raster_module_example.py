
"""
Rasterizing a mesh to a volumetric datastructure
"""

import numpy as np
import pandas as pd
import compas
from compas.datastructures import Mesh
import pyvista as pv

import volpy

__author__ = "Shervin Azadi, and Pirouz Nourian"
__copyright__ = "???"
__credits__ = ["Shervin Azadi", "Pirouz Nourian"]
__license__ = "???"
__version__ = "0.0.2"
__maintainer__ = "Shervin Azadi"
__email__ = "shervinazadi93@gmail.com"
__status__ = "Dev"

####################################################
# INPUTS
####################################################


vs = 0.04
voxel_size = np.array([vs, vs, vs])
tol = 1e-06
geo_mesh = Mesh.from_obj('Examples/IN/bunny.obj')

####################################################
# Rasterization
####################################################


volume, points = volpy.rasterization(
    geo_mesh, voxel_size, multi_core_process=False, return_points=True)

####################################################
# OUTPUTS
####################################################

vol_filepath = 'Examples/PY_OUT/bunny_volume.csv'
vol_metadata = pd.Series(
    [
        (f'voxel_size:{vs}-{vs}-{vs}'),
        ('name: bunny')
    ])

volpy.vol_to_csv(volume, vol_filepath, metadata=vol_metadata)


pnt_filepath = 'Examples/PY_OUT/bunny_voxels.csv'
pnt_metadata = pd.Series(
    [
        (f'voxel_size:{vs}-{vs}-{vs}'),
        ('name: bunny')
    ])

volpy.pnts_to_csv(points, pnt_filepath, metadata=pnt_metadata)


####################################################
# Visualization : PyVista
####################################################
