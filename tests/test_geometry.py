import topogenesis as tg
import numpy as np
import os

file_directory = os.path.dirname(os.path.abspath(__file__))
sample_data_path = os.path.join(os.path.dirname(file_directory), "data")


def test_surface_normal_newell_vectorized():
    """
    Testing the vectorized version of newell method for finding the normal of a triangle with hardcoded data
    """

    # Setup
    tri = np.array([[-0.00601774128, 0.130592465, 0.0237104725],
                    [-0.0866273791, 0.153729707, 0.0216472838],
                    [-0.0290798154, 0.125226036, 0.00471670832]])

    expected_normal = np.array([-0.24662338, -0.81206861,  0.52888702])

    # Exercise
    computed_normal = tg.geometry.surface_normal_newell_vectorized(tri)

    # Verify
    np.testing.assert_allclose(
        computed_normal, expected_normal, rtol=1e-6, atol=0)

    # Cleanup


def test_triangle_line_intersect():
    """
    Testing the Triangle Line Intersect with hardcoded data
    """

    # Setup
    tri = np.array([[-0.00601774128, 0.130592465, 0.0237104725],
                    [-0.0866273791, 0.153729707, 0.0216472838],
                    [-0.0290798154, 0.125226036, 0.00471670832]])
    line = np.array([[-0.28719836, -0.67555254, 0.54557851],
                     [0.20604841, 0.94858468, -0.51219553]])

    expected_point = np.array([-0.04057498,  0.13651607,  0.01669149])

    # Exercise
    computed_point = tg.geometry.triangle_line_intersect(line, tri)

    # Verify
    np.testing.assert_allclose(
        computed_point, expected_point, rtol=1e-6, atol=0)

    # Cleanup


def test_mesh_sampling():
    """
    Testing the mesh sampling with sample
    """

    # Setup
    tol = 1e-09

    mesh = tg.geometry.load_mesh(os.path.join(
        sample_data_path, "bunny_lowpoly.obj"))

    expected_sample_cloud = np.array(
        [[0.0064880164, 0.05,  0.05],
         [-0.009171068, 0.05,  0.05],
         [-0.05, 0.05,  0.0358800451],
         [0.05, 0.05,  0.0254832824],
         [0.05, 0.05, -0.0011166829],
         [-0.05, 0.15,  0.0113995458],
         [-0.05, 0.15, -0.0041858891],
         [-0.05, 0.05, -0.0079428066]]
    )
    #
    # Exercise
    computedted_sample_cloud = tg.geometry.mesh_sampling(mesh, 0.1, tol=tol)

    # Verify
    np.testing.assert_allclose(
        computedted_sample_cloud, expected_sample_cloud, rtol=1e-06, atol=0)

    # Cleanup


def test_flat_mesh_sampling():
    """
    Testing flat mesh sampling with sample
    """

    # Setup
    tol = 1e-09

    mesh_vertices = np.array([[0.0, 1.0, 0.0],
                              [1.0, 1.0, 0.0],
                              [1.0, 0.0, 0.0]])
    mesh_faces = np.array([[0, 1, 2]])
    mesh = (mesh_vertices, mesh_faces)

    expected_sample_cloud = np.array(
        [[0.45, 0.75, 0.],
         [0.75, 0.45, 0.],
         [0.75, 0.75, 0.]]
    )
    #
    # Exercise
    computedted_sample_cloud = tg.geometry.mesh_sampling(mesh, 0.3, tol=tol)

    # Verify
    np.testing.assert_allclose(
        computedted_sample_cloud, expected_sample_cloud, rtol=1e-06, atol=0)

    # Cleanup
