import volpy as vp
import numpy as np


def test_newell_surface_normal():
    """
    Testing the newell method for finding the normal of a triangle with hardcoded data
    """

    # Setup
    tri = np.array([[-0.00601774128, 0.130592465, 0.0237104725],
                    [-0.0866273791, 0.153729707, 0.0216472838],
                    [-0.0290798154, 0.125226036, 0.00471670832]])
    expected_norm = np.array([-0.24662338, -0.81206861,  0.52888702])

    # Exercise
    computed_norm = vp.surface_normal_newell(tri)

    # Verify
    np.testing.assert_allclose(
        computed_norm, expected_norm, rtol=1e-7, atol=0)

    # Cleanup
