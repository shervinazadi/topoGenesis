import volpy as vp
import numpy as np

# defining the bounds and number of points
bounds = np.array([[0, 0, 0], [3, 3, 3]])
count = 10

# scattering random points
pc = vp.scatter(bounds, count)

# regularizing random points into a lattice
l, test = pc.regularize([1, 1, 1])

print(type(pc))
print(pc)
print(type(l))
print(l)
