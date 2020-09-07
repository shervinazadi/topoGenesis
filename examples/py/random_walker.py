import topogenesis as tg
import numpy as np

# create a step one moore neighbourhood
s = tg.create_stencil("von_neumann", 1)

# set the center to x direction to 0, since we are working in 2d
s.set_index([1, 0, 0], 0)
s.set_index([-1, 0, 0], 0)

# assign the sum function
s.function = tg.sfunc.random_choice

# initiate the lattice
l = tg.lattice([[0, -2, -1], [0, 2, 1]], default_value=0, dtype=int)
l[0, 2, 1] += 1

# the id of voxels (0,1,2, ... n)
l_inds = l.indicies

print(l)
print(l_inds)
print(type(l_inds))
random_neighbour = l_inds.apply_stencil(s)
print(random_neighbour)
# # main iteration forloop
# for i in range(10):
#     print(l)

#     # apply the stencil on the lattice
#     neighbor_sum = l.apply_stencil(s)

#     # apply cellular automata rules

#     # turn off if less than 2 or more than 3
#     l *= (neighbor_sum >= 2) * (neighbor_sum <= 3)

#     # turn on if 3 neighbours are on
#     l[(neighbor_sum == 3)] = 1
