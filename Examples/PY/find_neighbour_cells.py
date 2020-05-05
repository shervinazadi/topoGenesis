import numpy as np
import itertools

# square grid of number with one-layer border of -1
size = 4
sqr = np.arange(size**2).reshape(size, size)
sqr_paded = np.pad(sqr, (1, 1), mode='constant', constant_values=(-1, -1))
sqr_val_ind = np.argwhere(sqr_paded.ravel() > -1).ravel()

# neighbourhood description
axis_order = (0, 1)
steps = 1
shifts = np.array(list(itertools.product([0, -1, 1], repeat=2)))
shift_steps = np.sum(np.absolute(shifts), axis=1)
chosen_shift_ind = np.argwhere(shift_steps <= steps).ravel()
shifts = shifts[chosen_shift_ind]

# finding neighbours
neighs_all = [np.roll(sqr_paded, shift, axis_order).ravel()
              for shift in shifts]

neighs_stacked = np.stack(neighs_all, axis=-1)
neighs_sqr = neighs_stacked[sqr_val_ind]

# print the result
print(neighs_sqr)
