

"""
Rasterizing a mesh to a volumetric datastructure
"""

import numpy as np
import pandas as pd
import compas
from compas.datastructures import Mesh
import sys
import pyvista as pv
np.set_printoptions(threshold=sys.maxsize)

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

dim_num = 3
vs = 0.01
voxel_size = np.array([vs, vs, vs])
tol = 1e-06
geo_mesh = Mesh.from_obj('Examples/IN/bunny.obj')

####################################################
# Initialize the volumetric array
####################################################

# retrieve the bounding box information
mesh_bb = np.array(geo_mesh.bounding_box())
mesh_bb_min = np.amin(mesh_bb, axis=0)
mesh_bb_max = np.amax(mesh_bb, axis=0)
mesh_bb_size = mesh_bb_max - mesh_bb_min

# find the minimum index in discrete space
mesh_bb_min_z3 = np.rint(mesh_bb_min / voxel_size).astype(int)
# calculate the size of voxelated volume
vol_size = np.ceil((mesh_bb_size / voxel_size)+1).astype(int)
# initiate the 3d array of voxel space called volume
vol = np.zeros(vol_size)

####################################################
# claculate the origin and direction of rays
####################################################

# retriev the voxel index for ray origins
hit_vol_ind = np.indices(vol_size)
vol_ind_trans = np.transpose(hit_vol_ind) + mesh_bb_min_z3
hit_vol_ind = np.transpose(vol_ind_trans)
ray_orig_ind = [np.concatenate(np.transpose(
    np.take(hit_vol_ind, 0, axis=d + 1))) for d in range(dim_num)]  # this line has a problem given the negative indicies are included now
ray_orig_ind = np.concatenate(tuple(ray_orig_ind), axis=0)

# retrieve the direction of ray shooting for each origin point
normals = np.identity(dim_num).astype(int)
ray_dir = [np.tile(normals[d], (np.take(vol, 0, axis=d).size, 1))
           for d in range(dim_num)]  # this line has a problem given the negative indicies are included now
ray_dir = np.concatenate(tuple(ray_dir), axis=0)

####################################################
# intersection
####################################################

hit_positions = []
# iterate over the faces
# for face in geo_mesh.faces():
faces = list(geo_mesh.faces())
face = faces[42]
print(face)
face_verticies_xyz = geo_mesh.face_coordinates(face)
"""if len(face_verticies_xyz) != 3:
    continue"""

face_verticies_xyz = np.array(face_verticies_xyz)

# project the ray origin
proj_ray_orig = ray_orig_ind * voxel_size * (1 - ray_dir)

# check if any coordinate of the projected ray origin is in betwen the max and min of the coordinates of the face
min_con = proj_ray_orig >= np.amin(face_verticies_xyz, axis=0)*(1 - ray_dir)
max_con = proj_ray_orig <= np.amax(face_verticies_xyz, axis=0)*(1 - ray_dir)
in_range_rays = np.all(min_con * max_con, axis=1)

# retrieve the ray indicies that are in range
in_rang_ind = np.argwhere(in_range_rays).flatten()
#print(np.amin(face_verticies_xyz, axis=0))
#print(np.amax(face_verticies_xyz, axis=0))
#print(np.c_[proj_ray_orig, min_con, max_con, in_range_rays])
sel = ray_orig_ind[in_rang_ind]- mesh_bb_min_z3

faces = np.array([3, 0, 1, 2])
surf = pv.PolyData(face_verticies_xyz, faces)

values = np.zeros(vol_size)
sel = np.transpose(sel)
#np.put(values, sel, 1)
values[sel[0], sel[1], sel[2]] = 1
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
grid.origin = mesh_bb_min_z3 * voxel_size - voxel_size*0.5
grid.spacing = voxel_size  # These are the cell sizes along each axis

# Add the data values to the cell data
grid.cell_arrays["values"] = values.flatten(
    order="F")  # Flatten the array!

# filtering
threshed = grid.threshold([0.9, 1.1])
outline = grid.outline()
# plot each face with a different color
p = pv.Plotter()
p.add_mesh(surf, color="red")
p.add_mesh(outline, color="k")
p.add_mesh(threshed, show_edges=True, color="white", opacity=0.7)
p.show()
"""
# iterate over the rays
for ray in in_rang_ind:

    # calc ray origin position: Z3 to R3
    orig_pos = ray_orig_ind[ray] * voxel_size
    # retrieve ray direction
    direction = ray_dir[ray]
    # calc the destination of ray (max distance that it needs to travel)
    # this line has a problem given the negative indicies are included now
    dest_pos = orig_pos + ray_dir[ray] * mesh_bb_size

    # intersction
    hit_pt = compas.geometry.intersection_line_triangle(
        (orig_pos, dest_pos), face_verticies_xyz, tol=tol)
    if hit_pt is not None:
        hit_positions.append(hit_pt)

####################################################
# convert hit positions into volumetric data
####################################################

# round the positions to find the closest voxel
hit_positions = np.array(hit_positions)

# R3 to Z3
hit_indicies = np.rint(hit_positions / voxel_size)

# remove repeated points
hit_unq_ind = np.unique(hit_indicies, axis=0)

# calculate volum indecies
hit_vol_ind = np.transpose(hit_unq_ind - mesh_bb_min_z3).astype(int)

####################################################
# OUTPUTS
####################################################

# Z3 to R3
hit_unq_pos = hit_unq_ind * voxel_size

# hit position to panada dataframe
hit_unq_pos_df = pd.DataFrame(
    {'PX': hit_unq_pos[:, 0],
     'PY': hit_unq_pos[:, 1],
     'PZ': hit_unq_pos[:, 2],
     })


# set values in the volumetric data
vol[hit_vol_ind[0], hit_vol_ind[1], hit_vol_ind[2]] = 1

# get the indicies of the voxels
vol_3d_ind = np.indices(vol.shape)
vol_3d_ind_flat = np.c_[vol_3d_ind[0].ravel(
), vol_3d_ind[1].ravel(), vol_3d_ind[2].ravel()]  # this can be done with transpose as well

# flatten the volume
vol_flat = vol.ravel()

# volume data to panda dataframe
vol_df = pd.DataFrame(
    {'IX': vol_3d_ind_flat[:, 0],
     'IY': vol_3d_ind_flat[:, 1],
     'IZ': vol_3d_ind_flat[:, 2],
     'value': vol_flat,
     })

# save to csv
hit_unq_pos_df.to_csv('PY_OUT/bunny_voxels.csv', index=True, float_format='%g')
vol_df.to_csv('PY_OUT/bunny_volume.csv', index=False, float_format='%g')
"""
