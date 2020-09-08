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

# assign the sum function
s.function = tg.sfunc.random_choice

# initiate the lattice
l = tg.lattice([[0, -3, -3], [0, 3, 3]], default_value=0, dtype=int)
l[0, 3, 3] += 1

# the id of voxels (0,1,2, ... n)
l_inds = l.indicies

# main iteration forloop
for i in range(50):
    click.clear()
    print(l, flush=True)

    # apply the stencil to the lattice
    random_neighbour = l_inds.apply_stencil(s, border_condition="roll")

    # convert the new positions to lattice indexes
    new_pos = np.array(np.unravel_index(random_neighbour[l == 1], l.shape))

    # apply the movements
    l *= 0
    l[new_pos[0], new_pos[1], new_pos[2]] = 1

    # sleep for 0.2 seconds
    sleep(.2)
