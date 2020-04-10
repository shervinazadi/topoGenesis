

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
# INPUTS
####################################################

# read the voxel 3-dimensional indicies
vol_3d_ind_flat = np.genfromtxt('IN/volume_sphere_bool.csv',
                                delimiter=',', skip_header=1, usecols=(0, 1, 2)).astype(int)
# read voxel valus
vol_flat = np.genfromtxt('IN/volume_sphere_bool.csv',
                         delimiter=',', skip_header=1, usecols=3).astype(int)
# find the olume shape from indicies
vol_shape = np.amax(vol_3d_ind_flat, axis=0) + 1

# reshape the 1d array to get 3d array
vol = vol_flat.reshape(vol_shape)

####################################################
# Finding the voxel neighbours for each voxel
####################################################

# the id of voxels in a flatten shape (0,1,2, ... n)
vol_flat_inds = np.arange(vol.size)

# removing the indecies that are not filled in the volume
vol_flat_inds = ((vol_flat_inds + 1) * vol_flat) - 1

# reshape the 1dimensional indicies of voxels into volume shape
vol_flat_inds_3d = vol_flat_inds.reshape(vol_shape)

# offset the volume with value -1
vol_paded = np.pad(vol, (1, 1), mode='constant', constant_values=(-1, -1))

# offset the 1-dimensional indicies of the voxels that is rshaped to volume shape with value -1
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

# print(cell_neighbors)
# print(neighbors[0])
####################################################
# Creating the graph
####################################################


"""
edges = []
for cell_neigh in cell_neighbors:
    cell = cell_neigh[0]
    for neigh in cell_neigh[1:]:
        if neigh != -1 and neigh > cell:
            edges.append((cell, neigh))
"""
connect_G = nx.Graph()
G.add_edges_from(edges)
