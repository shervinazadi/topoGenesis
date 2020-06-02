import volpy as vp
import numpy as np

# initialize a latice
l = vp.lattice(bounds=[[0, 0, 0], [5, 5, 5]])

# initialize a lattice with boolian or integer datatype
# l_bool = vp.lattice(shape=(3, 3, 3), dtype=bool)
# l_int = vp.lattice(shape=(3, 3, 3), dtype=int)

print(l)
