
"""
Algorithm Sketches that need to be integrated with the rest of library
"""

import itertools
import numpy as np
import pandas as pd
import networkx as nx
import compas
from compas.datastructures import Mesh
import concurrent.futures
import warnings
from topogenesis.data import find_neighbours

__author__ = "Shervin Azadi, and Pirouz Nourian"
__copyright__ = "???"
__credits__ = ["Shervin Azadi", "Pirouz Nourian"]
__license__ = "???"
__version__ = "0.0.2"
__maintainer__ = "Shervin Azadi"
__email__ = "shervinazadi93@gmail.com"
__status__ = "Dev"


def connectivity_graph(vol, steps):

    ####################################################
    # Finding the voxel neighbours for each voxel (generalized)
    ####################################################

    cell_neighbors = find_neighbours(vol, steps)

    ####################################################
    # Creating the graph
    ####################################################

    # the code in this section is a vectorized equivalent of:
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

    # tile the first column (cell id) with the size of the number of the neighs
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


def vol_to_pandas(vol):
    # get the indices of the voxels
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
    vol_df = vol_to_pandas(vol)
    # panada-dataframe to csv

    with open(filepath, 'w') as df_out:
        if metawrite:
            df_out.write('metadata:\n')
            metadata.to_csv(df_out, index=False,
                            header=False, float_format='%g')
            df_out.write('\n')
        df_out.write('volume:\n')
        vol_df.to_csv(df_out, index=False, float_format='%g')


def CellularAutomata(data):

    # convert string to json
    if isinstance(data, str):
        data = json.loads(data)

    # retrieve the position list
    voxel_positions = np.array(data['voxel_positions'])

    # construct the volume
    minbound = voxel_positions.min(axis=0)
    maxbound = voxel_positions.max(axis=0)
    volume = np.zeros(maxbound-minbound + 1).astype(int)

    # fill in the volume
    mapped_ind = voxel_positions - minbound
    volume[mapped_ind[:, 0], mapped_ind[:, 1], mapped_ind[:, 2]] = 1

    # pad the volume with zero in every direction
    volume = np.pad(volume, (1, 1), mode='constant', constant_values=(0, 0))

    # the id of voxels (0,1,2, ... n)
    volume_inds = np.arange(volume.size).reshape(volume.shape)

    # computing all the possible shifts to apply to the array
    shifts = np.array(list(itertools.product([0, -1, 1], repeat=3)))

    # gattering all the replacements in the columns
    replaced_columns = [
        np.roll(volume_inds, shift, np.arange(3)).ravel() for shift in shifts]

    # stacking the columns
    cell_neighbors = np.stack(replaced_columns, axis=-1)

    # replace neighbours by their value in volume
    volume_flat = volume.ravel()
    neighbor_values = volume_flat[cell_neighbors]

    # sum the neighbour values
    neighbor_sum = neighbor_values.sum(axis=1)

    # turn off if less than 2 neighbours on
    volume_flat *= (neighbor_sum >= 2)

    # turn off if more than 6 neighbours on
    volume_flat *= (neighbor_sum <= 6)

    # turn on if 3 neighbours are on
    volume_flat[(neighbor_sum >= 3) * (neighbor_sum <= 4)] = 1

    # on-cells 1D-index
    on_cells_1d_ind = np.argwhere(volume_flat == 1).ravel()

    # unravel the indices to 3D-index
    on_cells_3d_ind = np.stack(np.unravel_index(
        on_cells_1d_ind, volume.shape), axis=-1)

    # map back the indices (add initial minimum and subtract 1 for the padded layer)
    new_voxel_positions = on_cells_3d_ind + minbound - 1

    # package the results
    result = {
        'voxel_positions': new_voxel_positions.tolist(),
        'to_update': data['to_update'] + ['voxels'] if 'to_update' in data else ['voxels'],
    }

    # extend the data dictionary with the result dictionary
    data.update(result)

    return result


def RandomWalkingAgents(data):

    # convert string to json
    if isinstance(data, str):
        data = json.loads(data)

    # retrieve the position list
    voxel_positions = np.array(data['voxel_positions'])

    # construct the volume
    minbound = voxel_positions.min(axis=0)
    maxbound = voxel_positions.max(axis=0)
    volume = np.zeros(maxbound-minbound + 1).astype(int)

    # fill in the volume
    mapped_ind = voxel_positions - minbound
    volume[mapped_ind[:, 0], mapped_ind[:, 1], mapped_ind[:, 2]] = 1

    # pad the volume with zero in every direction
    volume = np.pad(volume, (1, 1), mode='constant', constant_values=(0, 0))

    # the id of voxels (0,1,2, ... n)
    volume_inds = np.arange(volume.size).reshape(volume.shape)

    # shifts to check: self + 6 neighbours of each voxel
    shifts = np.array([
        [0, 0, 0],  # self
        [1, 0, 0],  # left
        [-1, 0, 0],  # right
        [0, 1, 0],  # up
        [0, -1, 0],  # down
        [0, 0, 1],  # back
        [0, 0, -1],  # front
    ])

    # gattering all the replacements in the columns
    replaced_columns = [
        np.roll(volume_inds, shift, np.arange(3)).ravel() for shift in shifts]

    # stacking the columns
    cell_neighbors = np.stack(replaced_columns, axis=-1)

    # replace neighbours by their value in volume and flip it: 0:occupied, 1:empty
    volume_flat = volume.ravel()
    neighbor_values_flipped = 1-volume_flat[cell_neighbors]

    # multiply the cell neighbours by their flipped values to remove all occupied neighbours. +1 and -1 is to prevent mixing the empty cells and the cell id 0, so we mark the empty cells -1 instead of 0
    empty_neighbours = neighbor_values_flipped * (cell_neighbors + 1) - 1

    # extracting the id and flipped value of neighbours of the filled voxels: current position of agents
    agent_neighbour_values_flipped = neighbor_values_flipped[np.where(
        volume_flat)]
    agent_neighbour_id = cell_neighbors[np.where(volume_flat)]

    # assigning random value to each neighbour (this can later be specified by a field instead of randomvalues)
    rand_shape = agent_neighbour_values_flipped.shape
    agn_neigh_priority = agent_neighbour_values_flipped * \
        np.random.rand(rand_shape[0], rand_shape[1])

    # getting the argmax to find the selected neighbours
    neigh_selection = agn_neigh_priority.argmax(axis=1)

    # extracting the voxel id of the current position and the next selected position of all agents
    cur_pos_id = agent_neighbour_id[:, 0]
    sel_pos_id = agent_neighbour_id[np.arange(
        neigh_selection.size), neigh_selection]

    # checking for unique ids in the new_position_id: to prevent two agents merging together by moving into one voxel
    __, unq_indices = np.unique(sel_pos_id, return_index=True)

    # setting the unique voxels in the selected position to current position: updating the cur_pos_id. (this is computationally cheaper than setting it in a new array)
    cur_pos_id[unq_indices] = sel_pos_id[unq_indices]

    # emptying the volume and filling the selected neighbours
    volume_flat *= 0
    volume_flat[cur_pos_id] = 1

    # on-cells 1D-index
    on_cells_1d_ind = np.argwhere(volume_flat == 1).ravel()

    # unravel the indices to 3D-index
    on_cells_3d_ind = np.stack(np.unravel_index(
        on_cells_1d_ind, volume.shape), axis=-1)

    # map back the indices (add initial minimum and subtract 1 for the padded layer)
    new_voxel_positions = on_cells_3d_ind + minbound - 1

    ###########################################################################
    # ATTENTION:
    # Agents are not able to merge anymore. if two agent has chose one single
    # voxel as the next position. one of would randomly be selected to move to
    # the next position and the other one will remain in his old position
    ###########################################################################

    # package the results
    result = {
        # 'to_update': ["hi there"],
        'voxel_positions': new_voxel_positions.tolist(),
        'to_update': data['to_update'] + ['voxels'] if 'to_update' in data else ['voxels'],
    }

    # extend the data dictionary with the result dictionary
    data.update(result)

    return result


def gradient(sfield):

    # partial derivatives
    dx = (sfield[:-2, 1:-1, 1:-1] - sfield[2:, 1:-1, 1:-1]) * 0.5
    dy = (sfield[1:-1, :-2, 1:-1] - sfield[1:-1, 2:, 1:-1]) * 0.5
    dz = (sfield[1:-1, 1:-1, :-2] - sfield[1:-1, 1:-1, 2:]) * 0.5

    # stack gradient
    g = np.stack([dx, dy, dz], axis=0)

    return(g)


def divergence(vfield):

    X = vfield[0]
    Y = vfield[0]
    Z = vfield[0]

    # partial derivatives
    dx = (X[:-2, 1:-1, 1:-1] - X[2:, 1:-1, 1:-1]) * 0.5
    dy = (Y[1:-1, :-2, 1:-1] - Y[1:-1, 2:, 1:-1]) * 0.5
    dz = (Z[1:-1, 1:-1, :-2] - Z[1:-1, 1:-1, 2:]) * 0.5

    # sum divergence
    d = dx + dy + dz

    return(d)
