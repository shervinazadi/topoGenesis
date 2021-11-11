

"""
Geometric Intersection Algorithms 
"""

import numpy as np
import pyvista as pv
import itertools
import concurrent.futures
import warnings
import os

from ..datastructures import cloud

__author__ = "Shervin Azadi, and Pirouz Nourian"
__copyright__ = "???"
__credits__ = ["Shervin Azadi", "Pirouz Nourian"]
__license__ = "???"
__version__ = "0.0.2"
__maintainer__ = "Shervin Azadi"
__email__ = "shervinazadi93@gmail.com"
__status__ = "Dev"

file_directory = os.path.dirname(os.path.abspath(__file__))


def mesh_sampling(mesh, unit, tol=1e-06, **kwargs):
    """This algorithm samples a mesh based on unit size

    Args:
        geo_mesh ([COMPAS Mesh]): [description]
        unit ([numpy array]): [Unit represents the unit size in the 
        sampling grid. It needs to be one float value or an array-like
        with three elements. In case that a scalar is given it will 
        used for all three dimensions]
        tol ([type], optional): [description]. Defaults to 1e-06.

    Returns:
        [type]: [description]
    """
    ####################################################
    # INPUTS
    ####################################################

    unit = np.array(unit)
    if unit.size == 1:
        unit = np.array([unit, unit, unit])
    elif unit.size != 3:
        raise ValueError(
            """unit needs to have three elements representing
            the unit size for mesh sampling in three dimensions""")
    dim_num = unit.size
    multi_core_process = kwargs.get('multi_core_process', False)
    return_ray_origin = kwargs.get('return_ray_origin', False)
    return_ray_dir = kwargs.get('return_ray_dir', False)

    # compare voxel size and tolerance and warn if it is not enough
    if min(unit) * 1e-06 < tol:
        warnings.warn(
            """Warning! The tolerance for rasterization is not small 
            enough, it may result in faulty results or failure of
            rasterization.Trydecreasingthetolerance or scaling
            the geometry.""")

    ####################################################
    # Initialize the volumetric array
    ####################################################
    mesh_vertices, mesh_faces = mesh

    # retrieve the bounding box information
    mesh_bb_min = np.amin(mesh_vertices, axis=0)
    mesh_bb_max = np.amax(mesh_vertices, axis=0)
    mesh_bb_size = mesh_bb_max - mesh_bb_min

    # find the minimum index in discrete space
    mesh_bb_min_z3 = np.rint(mesh_bb_min / unit).astype(int)
    # calculate the size of voxelated volume
    vol_size = np.ceil((mesh_bb_size / unit)+1).astype(int)
    # initiate the 3d array of voxel space called volume
    vol = np.zeros(vol_size)

    ####################################################
    # compute the origin and direction of rays
    ####################################################

    # increasing the vol_size by one to accommodate for
    # shooting from corners
    vol_size_off = vol_size + 1

    # retrieve the voxel index for ray origins
    hit_vol_ind = np.indices(vol_size_off)
    vol_ind_trans = np.transpose(hit_vol_ind) + mesh_bb_min_z3
    hit_vol_ind = np.transpose(vol_ind_trans)

    # retrieve the ray origin indices
    ray_orig_ind = [np.take(hit_vol_ind, 0, axis=d + 1)
                    .transpose((1, 2, 0))
                    .reshape(-1, 3)
                    for d in range(dim_num)]
    ray_orig_ind = np.vstack(ray_orig_ind)

    # retrieve the direction of ray shooting for each origin point
    normals = np.identity(dim_num).astype(int)

    # tile(stamp) the X-ray direction with the (Y-direction
    # * Z-direction) . Then repeat this for all dimensions
    # TODO: this line has a problem given the negative indices
    # are included now
    ray_dir = [np.tile(normals[d],
                       (vol_size_off[(d + 1) % dim_num]
                        * vol_size_off[(d + 2) % dim_num], 1))
               for d in range(dim_num)]
    ray_dir = np.vstack(ray_dir)

    # project the ray origin + shift it with half of the voxel size
    # to move it to corners of the voxels
    ray_orig = ray_orig_ind * unit + unit * -0.5  # * (1 - ray_dir)

    # project the ray origin
    proj_ray_orig = ray_orig * (1 - ray_dir)

    ####################################################
    # intersection
    ####################################################

    samples = []

    # check if multiprocessing is allowed
    if multi_core_process:
        # open the context manager
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # submit the processes
            # results = [executor.submit(intersect, geo_mesh, face,
            # unit, mesh_bb_size, ray_orig, proj_ray_orig, ray_dir,
            # tol) for face in geo_mesh.faces()]
            results = [executor.submit(intersect, mesh_vertices[face], unit,
                                       mesh_bb_size, ray_orig, proj_ray_orig,
                                       ray_dir, tol) for face in mesh_faces]
            # fetch the results
            for f in concurrent.futures.as_completed(results):
                samples.extend(f.result())
    else:
        # iterate over the faces
        # for face in geo_mesh.faces():
        for face in mesh_faces:
            # print(face)
            # print(type(face))
            # id = np.array(face)
            # print(mesh_vertices[id])
            # print(mesh_vertices[np.array(face).astype(int)])
            # print(mesh_vertices[np.array(face)])

            face_hit_pos = intersect(mesh_vertices[face], unit, mesh_bb_size,
                                     ray_orig, proj_ray_orig, ray_dir, tol)
            samples.extend(face_hit_pos)

    ####################################################
    # OUTPUTS
    ####################################################

    # set the output list
    if len(samples) != 0:
        out = [cloud(np.array(samples))]
    else:
        out = [None]
    # check if the ray origins are requested
    if return_ray_origin:
        out.append(cloud(np.array(ray_orig)))
    # check if the ray direction are requested
    if return_ray_dir:
        out.append(ray_dir)

    # if the list has more than one item, return it as a tuple, if it
    # has only one item, return the item itself
    return tuple(out) if len(out) > 1 else out[0]


def intersect(face_vertices_xyz, unit, mesh_bb_size, ray_orig,
              proj_ray_orig, ray_dir, tol):
    face_hit_pos = []

    # check if the face is a triangle
    if face_vertices_xyz.shape[0] != 3:
        return([])

    # check if any coordinate of the projected ray origin is in betwen
    # the max and min of the coordinates of the face
    min_con = proj_ray_orig >= np.amin(
        face_vertices_xyz, axis=0)*(1 - ray_dir)
    max_con = proj_ray_orig <= np.amax(
        face_vertices_xyz, axis=0)*(1 - ray_dir)
    in_range_rays = np.all(min_con * max_con, axis=1)

    # retrieve the ray indices that are in range
    in_rang_ind = np.argwhere(in_range_rays).flatten()

    # iterate over the rays
    for ray in in_rang_ind:
        # retrieve ray direction
        direction = ray_dir[ray]
        # retrieve ray origin
        orig_pos = ray_orig[ray]
        # calc the destination of ray (max distance that it needs to
        # travel) this line has a problem given the negative indices
        # are included now
        # plus unit added in case of flat meshes
        dest_pos = orig_pos + direction * (mesh_bb_size + unit)

        # intersection
        # Translated from Pirouz C#
        hit_pt = triangle_line_intersect(
            (orig_pos, dest_pos), face_vertices_xyz, tol=tol)
        if hit_pt is not None:
            face_hit_pos.append(hit_pt)

    return(face_hit_pos)


def triangle_line_intersect(L, Vx, tol=1e-06):
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
    N = surface_normal_newell_vectorized(Vx)  # np.cross(U, V)

    Nomin = np.dot((O - L[0]), N)
    Denom = np.dot(N, (L[1] - L[0]))

    if Denom != 0:
        alpha = Nomin / Denom

        # L[0] + np.dot(alpha, (L[1] - L[0])): parameter along the
        # line where it intersects the plane in question, only if
        # not parallel to the plane
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
        normalized = n/norm

    return normalized
