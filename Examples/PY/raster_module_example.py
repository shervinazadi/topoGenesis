
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

vs = 0.01
voxel_size = np.array([vs, vs, vs])
tol = 1e-06
geo_mesh = Mesh.from_obj('Examples/IN/bunny.obj')

####################################################
# Rasterization
####################################################

volume, points = volpy.rasterization(
    geo_mesh, voxel_size, multi_core_process=True, return_points=True)

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

values = volume

# Create the spatial reference
grid = pv.UniformGrid()

# Set the grid dimensions: shape + 1 because we want to inject our values on
#   the CELL data
grid.dimensions = np.array(values.shape) + 1

# retrieve the bounding box information
mesh_bb = np.array(geo_mesh.bounding_box())
mesh_bb_min = np.amin(mesh_bb, axis=0)

# Edit the spatial reference
grid.origin = mesh_bb_min  # The bottom left corner of the data set
grid.spacing = voxel_size  # These are the cell sizes along each axis

# Add the data values to the cell data
grid.cell_arrays["values"] = values.flatten(order="F")  # Flatten the array!

# filtering
threshed = grid.threshold([0.5, 1.5])
outline = grid.outline()

# Now plot the grid!
# grid.plot(show_edges=True)

p = pv.Plotter()
p.add_mesh(outline, color="k")
p.add_mesh(threshed, show_edges=True, color="white")
p.show()
