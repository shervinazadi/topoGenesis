
"""
An example of exporting networkx graph model to csv fils
"""

import networkx as nx
import numpy as np
import pandas as pd
import math as m

__author__ = "Shervin Azadi"
__copyright__ = "???"
__credits__ = ["Shervin Azadi"]
__license__ = "???"
__version__ = "0.0.2"
__maintainer__ = "Shervin Azadi"
__email__ = "shervinazadi93@gmail.com"
__status__ = "Dev"

####################################################
# init an example graph
####################################################

size = 10
G = nx.Graph()
G.add_nodes_from([i for i in range(size)])
G.add_edges_from([(i, (i + 1) % size) for i in range(size)])

# add attributes to the nodes
node_bb = nx.betweenness_centrality(G)
nx.set_node_attributes(G, node_bb, 'betweenness')
names = {i: "node "+str(i) for i in range(size)}
nx.set_node_attributes(G, names, 'name')
pos = {i: (m.sin(3.14*2*i/size), m.cos(3.14*2*i/size), 0) for i in range(size)}
nx.set_node_attributes(G, pos, 'position')

# add attributes to the edges
edge_names = {(i, (i + 1) % size): "edge "+str(i) for i in range(size)}
nx.set_edge_attributes(G, edge_names, 'name')
weights = {(i, (i + 1) % size): 0.1*i for i in range(size)}
nx.set_edge_attributes(G, weights, 'weight')
edge_bb = nx.edge_betweenness_centrality(G, normalized=False)
nx.set_edge_attributes(G, edge_bb, 'betweenness')

####################################################
# retrieve the nodes in numpy array
####################################################
# retrieve node list
graph_nodes = np.array(G.nodes)

# retrieve attribs
graph_node_names = np.array(
    list(nx.get_node_attributes(G, 'name').items()))[:, 1]
graph_node_bbs = np.array(
    list(nx.get_node_attributes(G, 'betweenness').items()))[:, 1]
graph_node_pos = np.array([np.array(item[1]) for item in list(
    nx.get_node_attributes(G, 'position').items())])

# merge
graph_nodes_df = pd.DataFrame(
    {'node': graph_nodes,
     'name': graph_node_names,
     'betweenness': graph_node_bbs,
     'P.X': graph_node_pos[:, 0],
     'P.Y': graph_node_pos[:, 1],
     'P.Z': graph_node_pos[:, 2],
     })

####################################################
# retrieve the edges in numpy array
####################################################
# retrive edge list
graph_edges = np.array(G.edges)

# retrive attribs
graph_edge_names = np.array(
    list(nx.get_edge_attributes(G, 'name').items()))[:, 1]
graph_edge_weights = np.array(
    list(nx.get_edge_attributes(G, 'weight').items()))[:, 1]
graph_edge_bbs = np.array(
    list(nx.get_edge_attributes(G, 'betweenness').items()))[:, 1]

# merge
graph_edges = np.c_[graph_edges, graph_edge_names,
                    graph_edge_weights, graph_edge_bbs]
graph_edges_df = pd.DataFrame(
    {'str': graph_edges[:, 0],
     'end': graph_edges[:, 1],
     'name': graph_edge_names,
     'weight': graph_edge_weights,
     'betweenness': graph_edge_bbs,
     })

####################################################
# retrieve the adjacency matrix
####################################################

graph_adj_matrix = nx.to_numpy_array(G)
graph_adj_matrix_df = pd.DataFrame(data=graph_adj_matrix)

####################################################
# save to CSV
####################################################
graph_nodes_df.to_csv('Examples/PY_OUT/graph_nodes.csv',
                      index=True, float_format='%.3f')
graph_edges_df.to_csv('Examples/PY_OUT/graph_edges.csv',
                      index=True, float_format='%.3f')
graph_adj_matrix_df.to_csv(
    'Examples/PY_OUT/graph_adj_matrix.csv', index=True, float_format='%.3f')
