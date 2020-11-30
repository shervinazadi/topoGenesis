
"""
Cloud DataStructure
"""

import numpy as np
import pandas as pd
import pyvista as pv
import itertools
import concurrent.futures
import warnings
import os

from .datastructures import stencil, cloud, lattice, to_lattice

__author__ = "Shervin Azadi, and Pirouz Nourian"
__copyright__ = "???"
__credits__ = ["Shervin Azadi", "Pirouz Nourian"]
__license__ = "???"
__version__ = "0.0.2"
__maintainer__ = "Shervin Azadi"
__email__ = "shervinazadi93@gmail.com"
__status__ = "Dev"

file_directory = os.path.dirname(os.path.abspath(__file__))


def scatter(bounds, count):

    point_array = np.random.uniform(bounds[0],
                                    bounds[1],
                                    (count, bounds.shape[1]))
    return cloud(point_array)


def cloud_from_csv(file_path, delimiter=','):

    point_array = np.genfromtxt(file_path, delimiter=delimiter)
    return cloud(point_array)


def cloud_from_mesh_vertices(mesh_path):

    # load the mesh using pyvista
    pv_mesh = pv.read(mesh_path)

    # extract vertices
    v = np.array(pv_mesh.points)

    # return vertices as cloud
    return cloud(v)


# extra import function
# TODO: this function is updated make sure that it is functioning properly
def lattice_from_csv(file_path):
    # read metadata
    meta_df = pd.read_csv(file_path, nrows=3)

    shape = np.array(meta_df['shape'])
    unit = np.array(meta_df['unit'])
    minbound = np.array(meta_df['minbound'])

    # read lattice
    lattice_df = pd.read_csv(file_path, skiprows=5)

    # create the buffer
    buffer = np.array(lattice_df['value']).reshape(shape)

    # create the lattice
    l = to_lattice(buffer, minbound=minbound, unit=unit)

    return l


def marching_cube_vis(p, cube_lattice, style_str):

    # extract cube indices
    cube_ind = np.transpose(np.indices(cube_lattice.shape),
                            (1, 2, 3, 0)).reshape(-1, 3)
    # extract cube positions
    cube_pos = (cube_ind + 0.5) * cube_lattice.unit + cube_lattice.minbound

    # extract cube tid
    cube_tid = cube_lattice.ravel()

    # remove the cube position and tid where tid is 0
    filled_cube_pos = cube_pos[cube_tid > 0]
    filled_cube_tid = cube_tid[cube_tid > 0]

    if style_str != "chamfer":
        raise ValueError(
            "Meshing style is not valid. Valid styles are: ['chamfer']")

    # load tiles
    tiles = [0]
    for i in range(1, 256):
        tile_path = os.path.join(os.path.dirname(
            file_directory), "resources/mc_tiles", style_str, "Tile_{0:03d}.obj".format(i))
        tile = pv.read(tile_path)
        tile.points *= cube_lattice.unit
        tiles.append(tile)

    new_points = tiles[filled_cube_tid[0]].points + filled_cube_pos[0]
    new_faces = tiles[filled_cube_tid[0]].faces.reshape(-1, 4)

    # merge tiles
    for i in range(1, filled_cube_tid.size):
        tile = tiles[filled_cube_tid[i]]
        # add the faces list, changing the point numbers
        new_faces = np.concatenate(
            (new_faces, tile.faces.reshape(-1, 4) + np.array([0, 1, 1, 1])*new_points.shape[0]), axis=0)
        # add the new points, change the position based on the location
        new_points = np.concatenate(
            (new_points, tile.points + filled_cube_pos[i]), axis=0)

    # construct the new mesh and add it to plot
    new_tile = pv.PolyData(new_points, new_faces)
    p.add_mesh(new_tile, color='#abd8ff')

    return p
