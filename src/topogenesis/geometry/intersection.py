

"""
Geometric Intersection Algorithms 
"""

import numpy as np
import pyvista as pv
import itertools
import concurrent.futures
import warnings
import os

__author__ = "Shervin Azadi, and Pirouz Nourian"
__copyright__ = "???"
__credits__ = ["Shervin Azadi", "Pirouz Nourian"]
__license__ = "???"
__version__ = "0.0.2"
__maintainer__ = "Shervin Azadi"
__email__ = "shervinazadi93@gmail.com"
__status__ = "Dev"

file_directory = os.path.dirname(os.path.abspath(__file__))

def intersect(face_verticies_xyz, unit, mesh_bb_size, ray_orig, proj_ray_orig, ray_dir, tol):
    face_hit_pos = []

    # check if the face is a triangle
    if face_verticies_xyz.shape[0] != 3:
        return([])

    # check if any coordinate of the projected ray origin is in betwen the max and min of the coordinates of the face
    min_con = proj_ray_orig >= np.amin(
        face_verticies_xyz, axis=0)*(1 - ray_dir)
    max_con = proj_ray_orig <= np.amax(
        face_verticies_xyz, axis=0)*(1 - ray_dir)
    in_range_rays = np.all(min_con * max_con, axis=1)

    # retrieve the ray indicies that are in range
    in_rang_ind = np.argwhere(in_range_rays).flatten()

    # iterate over the rays
    for ray in in_rang_ind:
        # retrieve ray direction
        direction = ray_dir[ray]
        # retrieve ray origin
        orig_pos = ray_orig[ray]
        # calc the destination of ray (max distance that it needs to travel)
        # this line has a problem given the negative indicies are included now
        dest_pos = orig_pos + direction * mesh_bb_size

        # intersction
        # compas version
        # hit_pt = compas.geometry.intersection_line_triangle((orig_pos, dest_pos), face_verticies_xyz, tol=tol)
        # Translated from Pirouz C#
        hit_pt = TriangleLineIntersect(
            (orig_pos, dest_pos), face_verticies_xyz, tol=tol)
        if hit_pt is not None:
            face_hit_pos.append(hit_pt)

    return(face_hit_pos)


def TriangleLineIntersect(L, Vx, tol=1e-06):
    """
    Computing the intersection of a line with a triangle
    Algorithm from http://geomalgorithms.com/a06-_intersect-2.html
    C# implementation from https://github.com/Pirouz-Nourian/Topological_Voxelizer_CSharp/blob/master/Voxelizer_Functions.cs

    Args:
        L ([2d np array]): List of two points specified by their coordinates
        Vx ([2d np array]): List of three points specified by their coordinates
        tol ([float], optional): tolerance. Defaults to 1e-06.

    Raises:
        ValueError: If the triangle contains more than three vertices

    Returns:
        [np array]: [description]
    """

    ####################################################
    # INPUTS
    ####################################################

    if len(Vx) != 3:
        raise ValueError('A triangle needs to have three vertexes')
    
    ####################################################
    # PROCEDURE
    ####################################################

    # finding U & V vectors
    O = Vx[0]
    U = Vx[1] - Vx[0]
    V = Vx[2] - Vx[0]
    # finding normal vector
    N = surface_normal_newell_vectorized(Vx) # np.cross(U, V)

    Nomin = np.dot((O - L[0]), N)
    Denom = np.dot(N, (L[1] - L[0]))

    if Denom != 0:
        alpha = Nomin / Denom

        # L[0] + np.dot(alpha, (L[1] - L[0])): parameter along the line where it intersects the plane in question, only if not paralell to the plane
        W = L[0] + np.dot(alpha, (L[1] - L[0])) - O

        UU = np.dot(U, U)
        VV = np.dot(V, V)
        UV = np.dot(U, V)
        WU = np.dot(W, U)
        WV = np.dot(W, V)

        STDenom = UV**2 - UU * VV

        s = (UV * WV - VV * WU) / STDenom
        t = (UV * WU - UU * WV) / STDenom

    ####################################################
    # OUTPUTS
    ####################################################

        if s + tol >= 0 and t + tol >= 0 and s + t <= 1 + 2*tol:
            Point = O + s * U + t * V
            return Point
        else:
            return None
    else:
        return None


def surface_normal_newell_vectorized(poly):
    """    
    https://stackoverflow.com/questions/39001642/calculating-surface-normal-in-python-using-newells-method
    Newell Method explained here: https://www.researchgate.net/publication/324921216_Topology_On_Topology_and_Topological_Data_Models_in_Geometric_Modeling_of_Space

    Args:
        poly ([2d np array]): List of vertices specified by their coordinates 

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """

    # This section is the vectorized equivalent of this code

    """
    n = np.array([0.0, 0.0, 0.0])

    for i in range(3):
        j = (i+1) % 3
        n[0] += (poly[i][1] - poly[j][1]) * (poly[i][2] + poly[j][2])
        n[1] += (poly[i][2] - poly[j][2]) * (poly[i][0] + poly[j][0])
        n[2] += (poly[i][0] - poly[j][0]) * (poly[i][1] + poly[j][1])
    """

    poly_10 = np.roll(poly, [-1, 0], np.arange(2))
    poly_01 = np.roll(poly, [0, -1], np.arange(2))
    poly_11 = np.roll(poly, [-1, -1], np.arange(2))

    n = np.roll(np.sum((poly - poly_10) * (poly_01 + poly_11), axis=0), -1, 0)
   
    norm = np.linalg.norm(n)
    if norm == 0:
        raise ValueError('zero norm')
    else:
        normalised = n/norm

    return normalised
