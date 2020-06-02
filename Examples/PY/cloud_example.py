import volpy as vp
import numpy as np

bounds = np.array([[0, 0, 0], [1, 1, 1]])
count = 10
pc = vp.scatter(bounds, count)

print(type(pc))
