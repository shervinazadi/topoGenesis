import numpy as np
import volpy as vp

tri = np.array([[-0.00601774128, 0.130592465, 0.0237104725],
                [-0.0866273791, 0.153729707, 0.0216472838],
                [-0.0290798154, 0.125226036, 0.00471670832]])

# print(vp.surface_normal_newell(tri))
print(vp.surface_normal_newell_vectorized(tri))
