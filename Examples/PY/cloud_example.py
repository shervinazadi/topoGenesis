import volpy as vp
import numpy as np

# defining the bounds and number of points
bounds = np.array([[0, 0, 0], [3, 3, 3]])
count = 10

# scattering random points and making a point cloud
pc = vp.scatter(bounds, count)

print(type(pc))
print(pc)

# regularizing random points into a lattice
l = pc.regularize([1, 1, 1])


print(type(l))
print(l)
