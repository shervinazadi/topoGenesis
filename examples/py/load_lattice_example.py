import topogenesis as tg
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)
# in this example we need to:
# 1. load the bunny volume into a lattice
bunny_lattice = tg.lattice_from_csv('data/bunny_volume.csv')

# 2. define a stencil (methods implemented)
stencil = tg.create_stencil("von_neumann", 1, 1)
# print(stencil)
# print(type(stencil))

# locs = tg.expand_stencil(stencil)
# print(locs)
# print(type(locs))

# 3. derive the connectivity graph of the rasterized bunny with the help of stencil (graph constructor needs to be implemented)
neighs = tg.find_neighbours(bunny_lattice, stencil)
print(neighs)
