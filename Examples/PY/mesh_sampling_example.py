
"""
Rasterizing a mesh to a volumetric datastructure
"""
import os
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

vs = 0.002
voxel_size = np.array([vs, vs, vs])
tol = 1e-09
#dirpath=os.path.curdir
#geo_path = os.path.join(dirpath,os.path.commonpath('Examples/SampleData/bunny_lowpoly.obj'))
print('test'+'C:\\Users\\pnourian\\Documents\\GitHub\\VolPy\\src\\Examples\\SampleData\\bunny_lowpoly.obj')
geo_path='C:\\Users\\pnourian\\Documents\\GitHub\\VolPy\\Examples\\SampleData\\bunny_lowpoly.obj'#os.path.abspath(r'Examples\SampleData\bunny_lowpoly.obj')#r'Examples/SampleData/bunny_lowpoly.obj'
geo_mesh = ds.Mesh.from_obj(geo_path)

####################################################
# Rasterization
####################################################

volume, centroids, samples, ray_origins = vp.mesh_sampling(geo_mesh, 
                                            voxel_size, 
                                            multi_core_process=False, return_centroids=True, 
                                            return_samples=True, 
                                            return_ray_origin=True, 
                                            tol=tol)

####################################################
# OUTPUTS
####################################################
'''
# Save the volumetric data model
vol_filepath = 'Examples/SampleData/bunny_volume.csv'
vol_metadata = pd.Series(
    [
        (f'voxel_size-{vs}-{vs}-{vs}'),
        (f'volume_shape-{volume.shape[0]}-{volume.shape[1]}-{volume.shape[2]}'),
        ('name-bunny')
    ])

vp.vol_to_csv(volume, vol_filepath, metadata=vol_metadata)

# Save the point data model
pnt_filepath = 'Examples/SampleData/bunny_voxels.csv'
pnt_metadata = pd.Series(
    [
        (f'voxel_size:{vs}-{vs}-{vs}'),
        ('name: bunny')
    ])

vp.pnts_to_csv(centroids, pnt_filepath, metadata=pnt_metadata)

# Save the hitpoints to point data model
pnt_filepath = 'Examples/SampleData/bunny_hitpoints.csv'
pnt_metadata = pd.Series(
    [
        ('name: bunny hit points')
    ])

vp.pnts_to_csv(samples, pnt_filepath, metadata=pnt_metadata)
'''

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
mesh = pv.read(geo_path)

# Main Plotting:

# initiating the plotter
p = pv.Plotter()
# p.set_background([0.065, 0.065, 0.065]) # dark grey background
p.set_background([1.0, 1.0, 1.0]) # with background

# adding the boundingbox wireframe
p.add_mesh(outline, color="grey", label="Sampling Domain")

# adding the base mesh: light blue
#p.add_mesh(mesh, show_edges=True, color='#abd8ff', opacity=0.4, label="Base Mesh")

# adding the ray origins: dark blue
p.add_mesh(pv.PolyData(ray_origins), color='#004887', point_size=4, render_points_as_spheres=True, label="Ray Origins")

# adding the hit points: blue
p.add_mesh(pv.PolyData(samples), color='#2499ff',point_size=8, render_points_as_spheres=True, label="Intersection Points")

# adding the voxel centeroids: red
#p.add_mesh(pv.PolyData(centroids), color='#ff244c', point_size=15, render_points_as_spheres=True, label="Voxel Centroids")

# adding the voxels: light red
p.add_mesh(threshed, show_edges=True, color="#ff8fa3", opacity=1, label="Voxels")

# adding the legend
p.add_legend(bcolor=[0.9, 0.9, .9], border=True, size=[0.1, 0.1])

# Set a camera position
p.camera_position = [(0.25, 0.18, 0.5), (0, .1, 0), (0, 1, 0)]

# plotting
p.show()
