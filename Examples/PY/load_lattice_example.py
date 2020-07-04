import volpy as vp
import numpy as np

# in this example we need to:
# 1. load the bunny volume into a lattice (loading method needs to be implemented)
# 2. define a stencil (methods implemented)
# 3. derive the connectivity graph of the rasterized buny with the help of stencil (graph constructor needs to be implemented)

# initialize a latice
# l = vp.lattice(bounds=[[0, 0, 0], [3, 3, 3]], default_value=0, dtype=int)

bunny_lattice = vp.lattice_from_csv('Examples/SampleData/bunny_volume.csv')

# stencil = vp.create_stencil("von_neumann", 1, 1)
# print(stencil)
# print(type(stencil))

# locs = vp.expand_stencil(stencil)
# print(locs)
# print(type(locs))
