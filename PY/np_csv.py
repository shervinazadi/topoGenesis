import numpy as np

point_pos = np.random.rand(100, 3)

# Scale
# point_pos *= np.array([1, 2, 1])
# Translation
# point_pos += np.array([0, 2, 0])

print(point_pos)

np.savetxt("PY_OUT/point_pos.csv", point_pos, delimiter=",")
