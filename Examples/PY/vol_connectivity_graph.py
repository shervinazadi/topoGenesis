

"""
creating a connectivity graph from a boolean volume
"""

import itertools
import numpy as np
import pandas as pd
import networkx as nx

__author__ = "Shervin Azadi"
__copyright__ = "???"
__credits__ = ["Shervin Azadi"]
__license__ = "???"
__version__ = "0.0.1"
__maintainer__ = "Shervin Azadi"
__email__ = "shervinazadi93@gmail.com"
__status__ = "Dev"

####################################################
# ATTENTION!!!
####################################################
# Content of this file has een implemented in VolPy
# module, try using that module for more updated
# version of this algorithm
####################################################

####################################################
# INPUTS
####################################################

# read the voxel 3-dimensional indices
vol_3d_ind_flat = np.genfromtxt('IN/bunny_volume.csv',
                                delimiter=',', skip_header=1, usecols=(0, 1, 2)).astype(int)
# read voxel valus
vol_flat = np.genfromtxt('IN/bunny_volume.csv',
                         delimiter=',', skip_header=1, usecols=3).astype(int)
# find the olume shape from indices
vol_shape = np.amax(vol_3d_ind_flat, axis=0) + 1

# reshape the 1d array to get 3d array
vol = vol_flat.reshape(vol_shape)

####################################################
# Finding the voxel neighbours for each voxel (generalized)
####################################################

# the id of voxels in a flatten shape (0,1,2, ... n)
vol_flat_inds = np.arange(vol.size)

# removing the indecies that are not filled in the volume
vol_flat_inds = ((vol_flat_inds + 1) * vol_flat) - 1

# reshape the 1dimensional indices of voxels into volume shape
vol_flat_inds_3d = vol_flat_inds.reshape(vol_shape)

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
steps = 1
shift_steps = np.sum(np.absolute(shifts), axis=1)
chosen_shift_ind = np.argwhere(shift_steps <= steps).ravel()
shifts = shifts[chosen_shift_ind]

# gattering all the replacements in the collumns
replaced_columns = [np.roll(vol_flat_inds_3d_paded,
                            shift, np.arange(3)).ravel() for shift in shifts]

# stacking the columns and removing the pads (and also removing the neighbours of the empty voxels since we have tagged them -1 like paddings)
cell_neighbors = np.stack(replaced_columns, axis=-1)[origin_flat_ind]

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

# removing the index of the cell it self
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

# create graph and the edges
G = nx.Graph()
G.add_edges_from([tuple(edge) for edge in cell_edge_order.tolist()])

####################################################
# Creating the graph
####################################################

# retrieve node list
graph_nodes = np.array(G.nodes)
# retriev and combine voxel 3d indices
graph_nodes = np.c_[graph_nodes, vol_3d_ind_flat[graph_nodes]]

# merge
graph_nodes_df = pd.DataFrame(
    {'node': graph_nodes[:, 0],
     'IX': graph_nodes[:, 1],
     'IY': graph_nodes[:, 2],
     'IZ': graph_nodes[:, 3],
     })

# retrive edge list
graph_edges = np.array(G.edges)

# merge
graph_edges_df = pd.DataFrame(
    {'str': graph_edges[:, 0],
     'end': graph_edges[:, 1],
     })

# save csv
graph_nodes_df.to_csv('PY_OUT/vol_graph_nodes.csv',
                      index=False, float_format='%.3f')
graph_edges_df.to_csv('PY_OUT/vol_graph_edges.csv',
                      index=False, float_format='%.3f')
