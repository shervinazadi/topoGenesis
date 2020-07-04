import volpy as vp
import numpy as np

# initialize a latice
l = vp.lattice(bounds=[[0, 0, 0], [3, 3, 3]], default_value=0, dtype=int)


stencil = vp.create_stencil("von_neumann", 1, 1)
print(stencil)
print(type(stencil))

locs = vp.expand_stencil(stencil)
print(locs)
print(type(locs))
