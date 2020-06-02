import volpy as vp
import numpy as np

# initialize a latice
l = vp.lattice(bounds=[[0, 0, 0], [3, 3, 3]], default_value=0, dtype=int)

# initialize a lattice with boolian or integer datatype
# l_bool = vp.lattice(shape=(3, 3, 3), dtype=bool)
# l_int = vp.lattice(shape=(3, 3, 3), dtype=int)

print(f'lattice: {l}')
print(f'lattice bounds: {l.bounds}')
print(f'lattice unit: {l.unit}')
print(f'lattice minimum bound: {l.minbound}')
print(f'lattice maximum bound: {l.maxbound}')
