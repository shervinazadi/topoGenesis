import topogenesis as tg
import numpy as np
import os

file_directory = os.path.dirname(os.path.abspath(__file__))
sample_data_path = os.path.join(os.path.dirname(file_directory), "data")
np.random.seed(0)


def test_cellular_automata():
    """
    Testing the vectorized version of newell method for finding the normal of a triangle with hardcoded data
    """

    # Setup
    #######

    # initiate stencil
    s = tg.create_stencil("moore", 1)  # create a step one moore neighbourhood
    s.set_index([0, 0, 0], 0)  # set the center to 0
    s.function = tg.sfunc.sum  # assign the sum function
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

    # initiate lattice
    l = tg.lattice([[0, -1, -1], [0, 1, 1]], default_value=0, dtype=int)
    l[0, :, 1] += 1
    """
    print(l)
    [[[0 1 0]
    [0 1 0]
    [0 1 0]]]
    """

    expected_lattice = np.array([[[0, 0, 0],
                                  [1, 1, 1],
                                  [0, 0, 0]]])

    # Exercise
    ##########

    neighbor_sum = l.apply_stencil(s)  # apply the stencil on the lattice

    # apply cellular automata rules
    # turn off if less than 2 or more than 3
    l *= (neighbor_sum >= 2) * (neighbor_sum <= 3)
    l[(neighbor_sum == 3)] = 1  # turn on if 3 neighbours are on

    computed_latice = l

    # Verify
    np.testing.assert_allclose(
        computed_latice, expected_lattice, rtol=1e-6, atol=0)

    # Cleanup


def test_gradient_decent():
    """
    Testing the vectorized version of newell method for finding the normal of a triangle with hardcoded data
    """

    # Setup
    #######

    # initiate stencil

    # create a step one moore neighbourhood
    s = tg.create_stencil("von_neumann", 1)
    s.function = tg.sfunc.argmin  # assign the arg-minimum function

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

    expected_lattice = np.array([[[0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 1., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.]]])

    # Exercise
    ##########

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

    computed_latice = l_walk

    # Verify
    ########
    np.testing.assert_allclose(
        computed_latice, expected_lattice, rtol=1e-6, atol=0)

    # Cleanup
    #########


def test_boolean_marching_cubes():
    """
    Testing the vectorized version of newell method for finding the normal of a triangle with hardcoded data
    """

    # Setup
    #######

    # Specifing all the inputs
    vs = 0.05               # voxel size
    unit = [vs, vs, vs]     # unit size
    tol = 1e-09             # intersection tolerance
    mesh_path = os.path.relpath('data/bunny_lowpoly.obj')
    original_mesh = tg.geometry.load_mesh(mesh_path)

    expected_lattice = np.array([[[0, 128, 136,   8],
                                  [0, 160, 170,  10],
                                  [128, 168, 170,  10],
                                  [160,  42,  34,   2]],
                                 [[128, 200, 204,  12],
                                  [160, 250, 255,  15],
                                  [224, 254, 255,  15],
                                  [240, 191,  59,   3]],
                                 [[192, 204, 204,  12],
                                  [240, 255, 255,  15],
                                  [240, 255, 255,  15],
                                  [240, 255,  63,   3]],
                                 [[192, 204, 204,  12],
                                  [112, 247, 255,  15],
                                  [80, 117, 119,   7],
                                  [80,  85,  21,   1]],
                                 [[64,  68,  68,   4],
                                  [16,  81,  85,   5],
                                  [0,  16,  17,   1],
                                  [0,   0,   0,   0]]])

    # Exercise
    ##########

    # Sampling the mesh and constructing the point cloud
    sample_cloud = tg.geometry.mesh_sampling(original_mesh, unit, tol=tol)

    # Voxelating the point cloud to construct the lattice
    lattice = sample_cloud.voxelate(unit, closed=True)

    # Constructing the Cube Lattice using the Boolea Marching Cube Algorithm
    cube_lattice = lattice.boolean_marching_cubes()

    computed_latice = cube_lattice

    # Verify
    ########
    np.testing.assert_allclose(
        computed_latice, expected_lattice, rtol=1e-6, atol=0)

    # Cleanup
    #########


def test_abm_random_walker():
    """
    Testing the vectorized version of newell method for finding the normal of a triangle with hardcoded data
    """

    # Setup
    #######

    # initiate stencil

    # create a step one moore neighbourhood
    s = tg.create_stencil("von_neumann", 1)

    # set the center to 0, to prevent staying at the same point
    s.set_index([0, 0, 0], 0)

    # set the x-dimension to 0, since we are working in 2d
    s.set_index([1, 0, 0], 0)
    s.set_index([-1, 0, 0], 0)

    # assign the random choice function
    s.function = tg.sfunc.random_choice

    """
    print(s)
    [[[0 0 0]
    [0 0 0]
    [0 0 0]]

    [[0 1 0]
    [1 0 1]
    [0 1 0]]

    [[0 0 0]
    [0 0 0]
    [0 0 0]]]
    """

    # initiate the lattice 0x7x7
    l = tg.lattice([[0, -3, -3], [0, 3, 3]], default_value=0, dtype=int)

    # place the walker in the center of the lattice
    l[0, 3, 3] += 1

    """
    print(l)
    [[[0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0]
    [0 0 0 1 0 0 0]
    [0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0]]]
    """

    # retrieve the indices of cells (0,1,2, ... n)
    l_inds = l.indices

    """
    print(l_inds)
    [[[ 0  1  2  3  4  5  6]
    [ 7  8  9 10 11 12 13]
    [14 15 16 17 18 19 20]
    [21 22 23 24 25 26 27]
    [28 29 30 31 32 33 34]
    [35 36 37 38 39 40 41]
    [42 43 44 45 46 47 48]]]
    """

    expected_lattice = np.array([[[0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0]]])

    # Exercise
    ##########

    # apply the stencil (function) to the lattice
    random_neighbour = l_inds.apply_stencil(s, border_condition="roll")

    # convert the current positions id and selected neighbour id to lattice indices
    old_pos = np.array(np.unravel_index(l_inds[l > 0], l.shape))
    new_pos = np.array(np.unravel_index(random_neighbour[l > 0], l.shape))

    # apply the movements
    l[old_pos[0], old_pos[1], old_pos[2]] -= 1
    l[new_pos[0], new_pos[1], new_pos[2]] += 1

    computed_latice = l

    # Verify
    ########
    np.testing.assert_allclose(
        computed_latice, expected_lattice, rtol=1e-6, atol=0)

    # Cleanup
    #########
