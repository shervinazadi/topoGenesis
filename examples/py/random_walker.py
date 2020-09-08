import topogenesis as tg
import numpy as np
import click            # for cleaning the command line
from time import sleep  # for delaying between iterations

# create a step one moore neighbourhood
s = tg.create_stencil("von_neumann", 1)

# set the center to 0, to prevent staying at the same point
s.set_index([0, 0, 0], 0)

# set the x-dimension to 0, since we are working in 2d
s.set_index([1, 0, 0], 0)
s.set_index([-1, 0, 0], 0)

# assign the random choice function
s.function = tg.sfunc.random_choice

# initiate the lattice 0x7x7
l = tg.lattice([[0, -3, -3], [0, 3, 3]], default_value=0, dtype=int)

# place the walker in the center of the lattice
l[0, 3, 3] += 1

# retrieve the indicies of cells (0,1,2, ... n)
l_inds = l.indicies

# main iteration forloop
for i in range(30):

    # clear the print console
    click.clear()

    # print the state of the lattice
    print(l)

    # apply the stencil (function) to the lattice
    random_neighbour = l_inds.apply_stencil(s, border_condition="roll")

    # convert the current positions id and selected neighbour id to lattice indicies
    old_pos = np.array(np.unravel_index(l_inds[l > 0], l.shape))
    new_pos = np.array(np.unravel_index(random_neighbour[l > 0], l.shape))

    # apply the movements
    l[old_pos[0], old_pos[1], old_pos[2]] -= 1
    l[new_pos[0], new_pos[1], new_pos[2]] += 1

    # wait for 0.3 seconds
    sleep(.3)
