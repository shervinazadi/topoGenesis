import topogenesis as tg
import numpy as np

def test_newell_surface_normal_vectorized():
    """
    Testing the vectorized version of newell method for finding the normal of a triangle with hardcoded data
    """

    # Setup
    tri = np.array([[-0.00601774128, 0.130592465, 0.0237104725],
                    [-0.0866273791, 0.153729707, 0.0216472838],
                    [-0.0290798154, 0.125226036, 0.00471670832]])
    expected_normal = np.array([-0.24662338, -0.81206861,  0.52888702])

    # Exercise
    computed_normal = tg.surface_normal_newell_vectorized(tri)

    # Verify
    np.testing.assert_allclose(
        computed_normal, expected_normal, rtol=1e-6, atol=0)

    # Cleanup


def test_TriangleLineIntersect():
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
    computed_point = tg.TriangleLineIntersect(line, tri)

    # Verify
    np.testing.assert_allclose(
        computed_point, expected_point, rtol=1e-6, atol=0)

    # Cleanup
