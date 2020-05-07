
"""
Rasterizing a mesh to a volumetric datastructure
"""

import numpy as np
import pandas as pd
import compas
from compas.datastructures import Mesh
import concurrent.futures

__author__ = "Shervin Azadi, and Pirouz Nourian"
__copyright__ = "???"
__credits__ = ["Shervin Azadi", "Pirouz Nourian"]
__license__ = "???"
__version__ = "0.0.2"
__maintainer__ = "Shervin Azadi"
__email__ = "shervinazadi93@gmail.com"
__status__ = "Dev"


def raster_intersect(geo_mesh, face, voxel_size, mesh_bb_size, ray_orig_ind, ray_dir, tol):
    face_hit_pos = []
    face_verticies_xyz = geo_mesh.face_coordinates(face)
    if len(face_verticies_xyz) != 3:
        return([])

    face_verticies_xyz = np.array(face_verticies_xyz)

    # project the ray origin
    proj_ray_orig = ray_orig_ind * voxel_size * (1 - ray_dir)

    # check if any coordinate of the projected ray origin is in betwen the max and min of the coordinates of the face
    min_con = proj_ray_orig >= np.amin(
        face_verticies_xyz, axis=0)*(1 - ray_dir)
    max_con = proj_ray_orig <= np.amax(
        face_verticies_xyz, axis=0)*(1 - ray_dir)
    in_range_rays = np.all(min_con * max_con, axis=1)

    # retrieve the ray indicies that are in range
    in_rang_ind = np.argwhere(in_range_rays).flatten()

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
            face_hit_pos.append(hit_pt)

    return(face_hit_pos)


def rasterization(geo_mesh, voxel_size, tol=1e-06, **kwargs):

    ####################################################
    # INPUTS
    ####################################################

    dim_num = voxel_size.size
    multi_core_process = kwargs.get('multi_core_process', False)
    return_points = kwargs.get('return_points', False)

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

    # old way
    # ray_orig_ind = [np.concatenate(np.transpose(np.take(hit_vol_ind, 0, axis=d + 1))) for d in range(dim_num)]  # this line has a problem given the negative indicies are included now
    # ray_orig_ind = np.concatenate(tuple(ray_orig_ind), axis=0)
    # new way
    ray_orig_ind = [np.take(hit_vol_ind, 0, axis=d + 1).transpose((1,
                                                                   2, 0)).reshape(-1, 3) for d in range(dim_num)]
    ray_orig_ind = np.vstack(ray_orig_ind)

    # retrieve the direction of ray shooting for each origin point
    normals = np.identity(dim_num).astype(int)
    ray_dir = [np.tile(normals[d], (np.take(vol, 0, axis=d).size, 1)) for d in range(
        dim_num)]  # this line has a problem given the negative indicies are included now
    ray_dir = np.vstack(ray_dir)

    ####################################################
    # intersection
    ####################################################

    hit_positions = []

    # check if multiprocessing is allowed
    if multi_core_process:
        # open the context manager
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # submit the processes
            results = [executor.submit(
                raster_intersect, geo_mesh, face, voxel_size, mesh_bb_size, ray_orig_ind, ray_dir, tol) for face in geo_mesh.faces()]
            # fetch the results
            for f in concurrent.futures.as_completed(results):
                hit_positions.extend(f.result())
    else:
        # iterate over the faces
        for face in geo_mesh.faces():
            face_hit_pos = raster_intersect(geo_mesh, face, voxel_size, mesh_bb_size,
                                            ray_orig_ind, ray_dir, tol)
            hit_positions.extend(face_hit_pos)

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
    # hit_unq_pos = (hit_unq_ind - mesh_bb_min_z3) * voxel_size + mesh_bb_min
    hit_unq_pos = hit_unq_ind * voxel_size

    # set values in the volumetric data
    vol[hit_vol_ind[0], hit_vol_ind[1], hit_vol_ind[2]] = 1

    if return_points:
        return (vol, hit_unq_pos, hit_positions)  # return (vol, hit_unq_pos)
    else:
        return vol


def pnts_to_csv(pnts, filepath, **kwargs):

    metadata = kwargs.get('metadata', None)
    metawrite = False if metadata is None else True
    # point position to panada dataframe
    pnts_df = pd.DataFrame(
        {
            'PX': pnts[:, 0],
            'PY': pnts[:, 1],
            'PZ': pnts[:, 2],
        })

    # save to csv
    with open(filepath, 'w') as df_out:
        if metawrite:
            df_out.write('metadata:\n')
            metadata.to_csv(df_out, index=False,
                            header=False, float_format='%g')
            df_out.write('\n')
        df_out.write('points:\n')
        pnts_df.to_csv(df_out, index=True, float_format='%g')


def vol_to_panadas(vol):
    # get the indicies of the voxels
    vol_3d_ind = np.indices(vol.shape)

    # flatten except the last dimension
    vol_3d_ind_flat = vol_3d_ind.transpose(1, 2, 3, 0).reshape(-1, 3)

    # flatten the volume
    vol_flat = vol.ravel()

    # volume data to panda dataframe
    vol_df = pd.DataFrame(
        {'IX': vol_3d_ind_flat[:, 0],
            'IY': vol_3d_ind_flat[:, 1],
            'IZ': vol_3d_ind_flat[:, 2],
            'value': vol_flat,
         })
    return vol_df


def vol_to_csv(vol, filepath, **kwargs):

    metadata = kwargs.get('metadata', None)
    metawrite = False if metadata is None else True
    # volume to panda dataframe
    vol_df = vol_to_panadas(vol)
    # panada-dataframe to csv

    with open(filepath, 'w') as df_out:
        if metawrite:
            df_out.write('metadata:\n')
            metadata.to_csv(df_out, index=False,
                            header=False, float_format='%g')
            df_out.write('\n')
        df_out.write('volume:\n')
        vol_df.to_csv(df_out, index=False, float_format='%g')
