
"""
Rasterizing a mesh to a volumetric datastructure
"""

import numpy as np
import pandas as pd
import compas
import compas.datastructures as ds
from compas.geometry import Translation
import pyvista as pv

import volpy as vp

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
tol = 1e-09
geo_mesh = ds.Mesh.from_obj('Examples/IN/bunny.obj')

####################################################
# Rasterization
####################################################

volume, points, hits = vp.rasterization(
    geo_mesh, voxel_size, multi_core_process=True, return_points=True, tol=tol)

####################################################
# OUTPUTS
####################################################

# Save the volumetric data model
vol_filepath = 'Examples/PY_OUT/bunny_volume.csv'
vol_metadata = pd.Series(
    [
        (f'voxel_size:{vs}-{vs}-{vs}'),
        ('name: bunny')
    ])

vp.vol_to_csv(volume, vol_filepath, metadata=vol_metadata)

# Save the point data model
pnt_filepath = 'Examples/PY_OUT/bunny_voxels.csv'
pnt_metadata = pd.Series(
    [
        (f'voxel_size:{vs}-{vs}-{vs}'),
        ('name: bunny')
    ])

vp.pnts_to_csv(points, pnt_filepath, metadata=pnt_metadata)


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
# The bottom left corner of the data set
mesh_bb_min_rasterized = np.rint(mesh_bb_min / voxel_size) * voxel_size
grid.origin = mesh_bb_min_rasterized - voxel_size * 0.5
grid.spacing = voxel_size  # These are the cell sizes along each axis

# Add the data values to the cell data
grid.cell_arrays["values"] = values.flatten(order="F")  # Flatten the array!

# filtering the voxels
threshed = grid.threshold([0.9, 1.1])

# bounding box of the voxelation
outline = grid.outline()

# loading the base mesh
mesh = pv.read("Examples/IN/bunny.obj")

# Main Plotting:

# initiating the plotter
p = pv.Plotter()

# adding the base mesh
p.add_mesh(mesh, show_edges=True, color='white', opacity=0.3)

# adding the boundingbox wireframe
p.add_mesh(outline, color="k")

# adding the voxel centeroids
p.add_mesh(pv.PolyData(points), color='red',
           point_size=15, render_points_as_spheres=True)

# adding the hit points
p.add_mesh(pv.PolyData(hits), color='blue',
           point_size=12, render_points_as_spheres=True)

# adding the voxels
p.add_mesh(threshed, show_edges=True, color="white", opacity=0.5)

# plotting
p.show()
