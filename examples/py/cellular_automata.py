import topogenesis as tg
import numpy as np
import click            # for cleaning the command line
from time import sleep  # for delaying between iterations

# create a step one moore neighbourhood
s = tg.create_stencil("moore", 1)

# set the center to 0
s.set_index([0, 0, 0], 0)

# assign the sum function
s.function = tg.sfunc.sum  # np.sum

"""
print(s)
[[[1 1 1]
  [1 1 1]
  [1 1 1]]

 [[1 1 1]
  [1 0 1]
  [1 1 1]]

 [[1 1 1]
  [1 1 1]
  [1 1 1]]]
"""

# initiate the lattice
l = tg.lattice([[0, -1, -1], [0, 1, 1]], default_value=0, dtype=int)
l[0, :, 1] += 1

"""
print(l)
[[[0 1 0]
  [0 1 0]
  [0 1 0]]]
"""

# main iteration forloop
for i in range(10):

    # clear the print console
    click.clear()

    # print the state of the lattice
    print(l)

    # apply the stencil on the lattice
    neighbor_sum = l.apply_stencil(s)

    # apply cellular automata rules

    # turn off if less than 2 or more than 3
    l *= (neighbor_sum >= 2) * (neighbor_sum <= 3)

    # turn on if 3 neighbours are on
    l[(neighbor_sum == 3)] = 1

    # wait for 0.3 seconds
    sleep(.3)
