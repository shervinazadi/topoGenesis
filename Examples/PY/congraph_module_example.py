

"""
creating a connectivity graph from a boolean volume
"""

import itertools
import numpy as np
import pandas as pd
import networkx as nx
import topogenesis as tg

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

# read the voxel 3-dimensional indices
vol_3d_ind_flat = np.genfromtxt('Examples/SampleData/bunny_volume.csv',
                                delimiter=',', skip_header=1, usecols=(0, 1, 2)).astype(int)
# read voxel valus
vol_flat = np.genfromtxt('Examples/IN/bunny_volume.csv',
                         delimiter=',', skip_header=1, usecols=3).astype(int)

# find the olume shape from indices
vol_shape = np.amax(vol_3d_ind_flat, axis=0) + 1

# reshape the 1d array to get 3d array
vol = vol_flat.reshape(vol_shape)

####################################################
# Connectivity Graph
####################################################

CG = tg.connectivity_graph(vol, 3)

####################################################
# Save to CSV files
####################################################

# retrieve node list
graph_nodes = np.array(CG.nodes)

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
graph_edges = np.array(CG.edges)

# merge
graph_edges_df = pd.DataFrame(
    {'str': graph_edges[:, 0],
     'end': graph_edges[:, 1],
     })

# save csv
graph_nodes_df.to_csv('Examples/PY_OUT/vol_graph_nodes.csv',
                      index=False, float_format='%.3f')
graph_edges_df.to_csv('Examples/PY_OUT/vol_graph_edges.csv',
                      index=False, float_format='%.3f')
