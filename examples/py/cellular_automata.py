import topogenesis as tg
import numpy as np

# create a step one moore neighbourhood
s = tg.create_stencil("moore", 1)

# set the center to 0
s.set_index([0, 0, 0], 0)

# assign the sum function
s.function = np.sum

# initiate the lattice
l = tg.lattice([[0, -2, -1], [0, 2, 1]], default_value=0)
l[0, 1:4, 1] += 1

# main iteration forloop
for i in range(10):
    print(l)

    # apply the stencil on the lattice
    neighbor_sum = l.apply_stencil(s)

    # apply cellular automata rules

    # turn off if less than 2 or more than 3
    l *= (neighbor_sum >= 2) * (neighbor_sum <= 3)

    # turn on if 3 neighbours are on
    l[(neighbor_sum == 3)] = 1
