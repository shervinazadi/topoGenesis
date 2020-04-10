import numpy as np
import compas
from compas.datastructures import Mesh

####################################################
# INPUTS
####################################################

dim_num = 3
vs = 0.05
voxel_size = np.array([vs, vs, vs])
tol = np.min(voxel_size * 0.01)
geo_mesh = Mesh.from_obj('IN/bunny.obj')

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
vol_size = np.ceil(mesh_bb_size / voxel_size).astype(int)
# initiate the 3d array of voxel space called volume
vol = np.zeros(vol_size)

####################################################
# claculate the origin and direction of rays
####################################################

# retriev the voxel index for ray origins
vol_ind = np.indices(vol_size)
vol_ind_trans = np.transpose(vol_ind) + mesh_bb_min_z3
vol_ind = np.transpose(vol_ind_trans)
ray_orig_ind = [np.concatenate(np.transpose(
    np.take(vol_ind, 0, axis=d + 1))) for d in range(dim_num)]
ray_orig_ind = np.concatenate(tuple(ray_orig_ind), axis=0)

# retrieve the direction of ray shooting for each origin point
normals = np.identity(dim_num).astype(int)
ray_dir = [np.tile(normals[d], (np.take(vol, 0, axis=d).size, 1))
           for d in range(dim_num)]
ray_dir = np.concatenate(tuple(ray_dir), axis=0)

####################################################
# intersection
####################################################


hit_positions = []
for face in geo_mesh.faces():
    face_verticies_xyz = geo_mesh.face_coordinates(face)
    if len(face_verticies_xyz) != 3:
        continue

    # iterate over the rays
    for r in range(ray_orig_ind.shape[0]):
        # calc ray origin position: Z3 to R3
        orig_pos = ray_orig_ind[r] * voxel_size
        # retrieve ray direction
        direction = ray_dir[r]
        # calc the destination of ray (max distance that it needs to travel)
        dest_pos = orig_pos + ray_dir[r] * mesh_bb_size

        # intersction
        hit_pt = compas.geometry.intersection_line_triangle(
            (orig_pos, dest_pos), face_verticies_xyz, tol=1e-06)
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
vol_ind = np.transpose(hit_unq_ind - mesh_bb_min_z3).astype(int)

####################################################
# OUTPUTS
####################################################

# Z3 to R3
hit_unq_pos = hit_unq_ind * voxel_size

# set values in the volumetric data
vol[vol_ind[0], vol_ind[1], vol_ind[2]] = 1
