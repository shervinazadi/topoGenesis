import topogenesis as tg
import numpy as np
import os

file_directory = os.path.dirname(os.path.abspath(__file__))
sample_data_path = os.path.join(os.path.dirname(file_directory), "data")


def test_cloud_voxelation():
    """
    Testing voxelation method of cloud object by sample data
    """

    # Setup
    cloud = tg.cloud_from_csv(os.path.join(sample_data_path, "rdam_cloud.csv"))

    expected_lattice = tg.lattice_from_csv(
        os.path.join(sample_data_path, "rdam_lattice.csv"))

    # Exercise
    computed_lattice = cloud.voxelate(1, closed=True)

    # Verify
    np.testing.assert_allclose(
        expected_lattice, computed_lattice, rtol=1e-6, atol=0)

    # Cleanup


def test_lattice_centroid():
    """
    Testing centroid method of lattice object by sample data
    """

    # Setup
    lattice = tg.lattice_from_csv(os.path.join(
        sample_data_path, "rdam_lattice.csv"))

    expected_centroids = tg.cloud_from_csv(
        os.path.join(sample_data_path, "rdam_centroids.csv"))

    # Exercise
    computed_centroids = lattice.centroids

    # Verify
    np.testing.assert_allclose(
        expected_centroids, computed_centroids, rtol=1e-6, atol=0)

    # Cleanup
