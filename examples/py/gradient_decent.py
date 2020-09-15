from time import sleep  # for delaying between iterations
import click            # for cleaning the command line
import topogenesis as tg
import numpy as np
np.random.seed(0)

# create a step one moore neighbourhood
s = tg.create_stencil("von_neumann", 1)


# assign the arg-minimum function
s.function = tg.sfunc.argmin

"""
print(s)
[[[0 0 0]
  [0 1 0]
  [0 0 0]]

 [[0 1 0]
  [1 1 1]
  [0 1 0]]

 [[0 0 0]
  [0 1 0]
  [0 0 0]]]
"""

# initialize a 2d lattice with random values
r = np.random.rand(1, 5, 5)
l_vals = tg.to_lattice(r, [0, 0, 0])

"""
print(l_vals)
[[[0.5488135  0.71518937 0.60276338 0.54488318 0.4236548 ]
  [0.64589411 0.43758721 0.891773   0.96366276 0.38344152]
  [0.79172504 0.52889492 0.56804456 0.92559664 0.07103606]
  [0.0871293  0.0202184  0.83261985 0.77815675 0.87001215]
  [0.97861834 0.79915856 0.46147936 0.78052918 0.11827443]]]
"""

# initialize walkers lattice
z = np.zeros((1, 5, 5))
l_walk = tg.to_lattice(z, [0, 0, 0])
l_walk[0, 2, 2] += 1

"""
print(l_walk)
[[[0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0.]
  [0. 0. 1. 0. 0.]
  [0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0.]]]
"""

# retrieve lattice indices
l_inds = l_vals.indices

"""
print(l_inds)
[[[ 0  1  2  3  4]
  [ 5  6  7  8  9]
  [10 11 12 13 14]
  [15 16 17 18 19]
  [20 21 22 23 24]]]
"""

# main iteration forloop
for i in range(20):

    # clear the print console
    click.clear()

    # print the state of the lattice
    print(l_walk)
    print(l_vals)

    # apply the stencil (function) to the lattice
    local_min_neighbour = l_vals.arg_apply_stencil(
        l_inds, s, border_condition="pad_outside", padding_value=1.0)

    # convert the current positions id and selected neighbour id to lattice indices
    old_pos = np.array(np.unravel_index(l_inds[l_walk > 0], l_walk.shape))
    new_pos = np.array(np.unravel_index(
        local_min_neighbour[l_walk > 0], l_walk.shape))

    # apply the movements
    l_walk[old_pos[0], old_pos[1], old_pos[2]] -= 1
    l_walk[new_pos[0], new_pos[1], new_pos[2]] += 1

    # wait for 0.3 seconds
    sleep(.3)
