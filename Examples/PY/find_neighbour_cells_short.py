import numpy as np

# square grid of number with one-layer border of -1
size = 4
sqr = np.arange(size**2).reshape(size, size)

# neighbourhood description
axis_order = (0, 1)
shifts = [[0, 0],
          [0, 1],
          [0, -1],
          [1, 0],
          [-1, 0]]

# finding neighbours
neighs_all = [np.roll(sqr, shift, axis_order).ravel() for shift in shifts]

neighs_stacked = np.stack(neighs_all, axis=-1)

# print the main square
print(sqr)

# print the result
print(neighs_stacked)
