
"""
Rasterizing a mesh to a volumetric datastructure
"""

import itertools
import numpy as np
import pandas as pd
import networkx as nx
import compas
from compas.datastructures import Mesh
import concurrent.futures
import warnings

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

    # compare voxel size and tolerance and warn if it is not enough
    if min(voxel_size) * 1e-06 < tol:
        warnings.warn(
            "Warning! The tolerance for rasterization is not small enough, it may result in faulty results or failure of rasterization. Try decreasing the tolerance or scaling the geometry.")

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


def find_neighbours(vol, steps):

    # flatten the array
    vol_flat = vol.ravel()

    # the id of voxels in a flatten shape (0,1,2, ... n)
    vol_flat_inds = np.arange(vol.size)

    # removing the indecies that are not filled in the volume
    vol_flat_inds = ((vol_flat_inds + 1) * vol_flat) - 1

    # reshape the 1dimensional indices of voxels into volume shape
    vol_flat_inds_3d = vol_flat_inds.reshape(vol.shape)

    # offset the volume with value -1
    vol_paded = np.pad(vol, (1, 1), mode='constant', constant_values=(-1, -1))

    # offset the 1-dimensional indices of the voxels that is rshaped to volume shape with value -1
    vol_flat_inds_3d_paded = np.pad(vol_flat_inds_3d, (1, 1), mode='constant',
                                    constant_values=(-1, -1))

    vol_flat_inds_3d_paded_flat = vol_flat_inds_3d_paded.reshape(
        vol_flat_inds_3d_paded.size)

    # index of padded cells in flatten
    origin_flat_ind = np.argwhere(vol_flat_inds_3d_paded_flat != -1).ravel()

    # claculating all the possible shifts to apply to the array
    shifts = np.array(list(itertools.product([0, -1, 1], repeat=3)))

    # the number of steps that the neighbour is appart from the cell (setp=1 : 6 neighbour, step=2 : 18 neighbours, step=3 : 26 neighbours)
    shift_steps = np.sum(np.absolute(shifts), axis=1)
    chosen_shift_ind = np.argwhere(shift_steps <= steps).ravel()
    shifts = shifts[chosen_shift_ind]

    # gattering all the replacements in the collumns
    replaced_columns = [np.roll(vol_flat_inds_3d_paded,
                                shift, np.arange(3)).ravel() for shift in shifts]

    # stacking the columns and removing the pads (and also removing the neighbours of the empty voxels since we have tagged them -1 like paddings)
    cell_neighbors = np.stack(replaced_columns, axis=-1)[origin_flat_ind]

    return cell_neighbors


def connectivity_graph(vol, steps):

    ####################################################
    # Finding the voxel neighbours for each voxel (generalized)
    ####################################################

    cell_neighbors = find_neighbours(vol, steps)

    ####################################################
    # Creating the graph
    ####################################################

    # the code in this section is a vectorised equivalent of:
    """
    edges = []
    for cell_neigh in cell_neighbors:
        cell = cell_neigh[0]
        for neigh in cell_neigh[1:]:
            if neigh != -1 and neigh > cell:
                edges.append((cell, neigh))
    """

    # removing the index of the cell itself
    cell_only_neighs = cell_neighbors[:, 1:]

    # tile the first collumn (cell id) with the size of the number of the neighs
    cell_only_neighs_rind = np.tile(
        cell_neighbors[:, 0].reshape(-1, 1), (1, cell_only_neighs.shape[1]))

    # flatten the neighs and row indices
    neighs_flat = cell_only_neighs.ravel()
    neighs_row_ind_flat = cell_only_neighs_rind.ravel()

    # find th index of the neighbours with real indices (everything except -1)
    real_neigh_ind = np.argwhere(neighs_flat != -1).ravel()

    # stack to list together and transpose them
    cell_edge_order_trans = np.stack(
        [neighs_row_ind_flat[real_neigh_ind], neighs_flat[real_neigh_ind]])
    cell_edge_order = np.transpose(cell_edge_order_trans)

    # create graph based on the edges
    G = nx.Graph()
    G.add_edges_from([tuple(edge) for edge in cell_edge_order.tolist()])

    return G


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
