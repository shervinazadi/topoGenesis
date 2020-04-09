import networkx as nx
import numpy as np

size = 10

G = nx.Graph()
G.add_nodes_from([i for i in range(size)])
G.add_edges_from([(i, (i + 1) % size) for i in range(size)])

graph_nodes = np.array(G.nodes)
graph_edges = np.array(G.edges)

graph_nodes = np.stack((np.arange(size), graph_nodes), axis=-1)

np.savetxt("PY_OUT/graph_nodes.csv", graph_nodes, fmt='%i', delimiter=",")
np.savetxt("PY_OUT/graph_edges.csv", graph_edges, fmt='%i', delimiter=",")
