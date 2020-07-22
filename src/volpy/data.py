import numpy as np
import pyvista as pv
import itertools
import pandas as pd
import networkx as nx
import compas
from compas.datastructures import Mesh
import concurrent.futures
import warnings

class lattice(np.ndarray):

    def __new__(subtype, bounds, unit=1, dtype=float, buffer=None, offset=0,
                strides=None, order=None, default_value=None):

        # extracting min and max from bound and discrtizing it
        bounds = np.array(bounds)
        minbound = np.rint(bounds[0] / unit).astype(int)
        maxbound = np.rint(bounds[1] / unit).astype(int)
        bounds = np.array([minbound, maxbound])*unit

        # unit nparray
        unit = np.array(unit)

        # raise value error if the size of unit is neighter 1 nor the length of the minimum
        if unit.size != 1 and unit.size != minbound.size:
            raise ValueError(
                'the length of unit array needs to be either 1 or equal to the min/max arrays')

        # calculating shape based on bounds and unit
        shape = 1 + maxbound - minbound

        # set defualt value
        if default_value != None:
            buffer = np.tile(
                default_value, shape)
            #obj = obj * 0 + default_value

        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to lattice.__array_finalize__
        obj = super(lattice, subtype).__new__(subtype, shape, dtype,
                                              buffer, offset, strides,
                                              order)

        # set the  'bounds' attribute
        obj.bounds = bounds
        # set the attribute 'unit' to itself if it has the same size as the minimum,
        # if the size is 1, tile it with the size of minimum vector
        obj.unit = unit if unit.size == minbound.size else np.tile(
            unit, minbound.size)

        # init an empty connectivity
        obj.connectivity = None
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(lattice, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. lattice():
        #    obj is None
        #    (we're in the middle of the lattice.__new__
        #    constructor, and self.bounds will be set when we return to
        #    lattice.__new__)
        if obj is None:
            return
        # From view casting - e.g arr.view(lattice):
        #    obj is arr
        #    (type(obj) can be lattice)
        # From new-from-template - e.g lattice[:3]
        #    type(obj) is lattice
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'bounds', because this
        # method sees all creation of default objects - with the
        # lattice.__new__ constructor, but also with
        # arr.view(lattice).
        self.bounds = getattr(obj, 'bounds', None)
        self.bounds = getattr(obj, 'bounds', None)
        self.unit = getattr(obj, 'unit', None)
        self.connectivity = getattr(obj, 'connectivity', None)
        # We do not need to return anything

    @property
    def minbound(self):
        return self.bounds[0]

    @property
    def maxbound(self):
        return self.bounds[1]

    @property
    def centroids(self):
        # extract the indicies of the True values # with sparse matrix we dont need to search
        point_array = np.argwhere(self == True)
        # move to minimum
        point_array += self.minbound
        # convert to float
        point_array = point_array.astype(float)
        # return as a point cloud
        return cloud(point_array, dtype=float)

    def fast_vis(self, plot, show_outline=True, show_centroids=True):

        # Set the grid dimensions: shape + 1 because we want to inject our values on the CELL data
        grid = pv.UniformGrid()
        grid.dimensions = np.array(self.shape) + 1
        # The bottom left corner of the data set
        grid.origin = self.minbound - self.unit * 0.5
        grid.spacing = self.unit  # These are the cell sizes along each axis
        # Add the data values to the cell data
        grid.cell_arrays["values"] = self.flatten(
            order="F").astype(float)  # Flatten the array!
        # filtering the voxels
        threshed = grid.threshold([0.9, 1.1])

        # adding the voxels: light red
        plot.add_mesh(threshed, show_edges=True, color="#ff8fa3",
                      opacity=0.3, label="Cells")

        if show_outline:
            # adding the boundingbox wireframe
            plot.add_mesh(grid.outline(), color="grey", label="Domain")

        if show_centroids:
            # adding the voxel centeroids: red
            plot.add_mesh(pv.PolyData(self.centroids), color='#ff244c', point_size=5,
                          render_points_as_spheres=True, label="Cell Centroidss")

        return plot

    def find_connectivity(self, stencil):
        pass


class cloud(np.ndarray):

    def __new__(subtype, point_array, dtype=float, buffer=None, offset=0,
                strides=None, order=None):

        # extracting the shape from point_array
        shape = point_array.shape
        # using the point_array as the buffer
        buffer = point_array.flatten(order="C")

        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to cloud.__array_finalize__
        obj = super(cloud, subtype).__new__(subtype, shape, dtype,
                                            buffer, offset, strides,
                                            order)

        # set the  'bounds' attribute
        obj.bounds = np.array([obj.min(axis=0), obj.max(axis=0)])

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(cloud, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. cloud():
        #    obj is None
        #    (we're in the middle of the cloud.__new__
        #    constructor, and self.bounds will be set when we return to
        #    cloud.__new__)
        if obj is None:
            return
        # From view casting - e.g arr.view(cloud):
        #    obj is arr
        #    (type(obj) can be cloud)
        # From new-from-template - e.g cloud[:3]
        #    type(obj) is cloud
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'bounds', because this
        # method sees all creation of default objects - with the
        # cloud.__new__ constructor, but also with
        # arr.view(cloud).
        self.bounds = getattr(obj, 'bounds', None)
        # We do not need to return anything

    @property
    def minbound(self):
        return self.bounds[0]

    @property
    def maxbound(self):
        return self.bounds[1]

    def regularize(self, unit):
        """[summary]

        Arguments:
            unit {[float or array of floats]} -- [the unit separation between cells of lattice]

        Raises:
            ValueError: [unit needs to be either a float or an array of floats that has the same dimension of the points in the point cloud]

        Returns:
            [lattice] -- [a boolian latice representing the rasterization of point cloud]
        """
        unit = np.array(unit)
        if unit.size != 1 and unit.size != self.bounds.shape[1]:
            raise ValueError(
                'the length of unit array needs to be either 1 or equal to the dimension of point cloud')
        elif unit.size == 1:
            unit = np.tile(unit, (1, self.bounds.shape[1]))

        # finding the closest voxel to each point
        vox_ind = np.rint(self / unit)
        # removing repetitions
        unique_vox_ind = np.unique(vox_ind, axis=0).astype(int)
        # mapping the voxel indicies to real space
        reg_pnt = unique_vox_ind * unit

        # initializing the volume
        l = lattice([self.minbound, self.maxbound], unit=unit,
                    dtype=bool, default_value=False)

        # mapp the indicies to start from zero
        mapped_ind = unique_vox_ind - l.bounds[0]

        # setting the occupied voxels to True
        l[mapped_ind[:, 0], mapped_ind[:, 1], mapped_ind[:, 2]] = True

        return l

    def fast_vis(self, plot):

        # adding the original point cloud: blue
        plot.add_mesh(pv.PolyData(self), color='#2499ff',
                      point_size=3, render_points_as_spheres=True, label="Original Point Cloud")

        return plot


def expand_stencil(stencil):
    locations = np.argwhere(stencil) - (stencil.shape[0] - 1) / 2
    # calculating the distance of each neighbour
    sums = np.abs(locations).sum(axis=1)
    # sorting to identify the main cell
    order = np.argsort(sums)
    # sort and return
    return locations[order].astype(int)


def create_stencil(type_str, steps, clip=None):
    # check if clip is specified. if it is not, set it to the steps
    if clip == None:
        clip = steps
    # von neumann neighborhood
    if type_str == "von_neumann":
        # https://en.wikipedia.org/wiki/Von_Neumann_neighborhood
        # claculating all the possible shifts to apply to the array
        shifts = np.array(list(itertools.product(
            list(range(-clip, clip+1)), repeat=3)))

        # the number of steps that the neighbour is appart from the cell (setp=1 : 6 neighbour, step=2 : 18 neighbours, step=3 : 26 neighbours)
        shift_steps = np.sum(np.absolute(shifts), axis=1)
        # check the number of steps
        chosen_shift_ind = np.argwhere(shift_steps <= steps).ravel()
        # select the valid indices from shifts variable, transpose them to get separate indicies in rows, add the number of steps to make this an index
        locs = np.transpose(shifts[chosen_shift_ind]) + clip

        stencil = np.zeros((clip*2+1, clip*2+1, clip*2+1)).astype(int)
        stencil[locs[0], locs[1], locs[2]] = 1
    elif type_str == "moore":
        # https://en.wikipedia.org/wiki/Moore_neighborhood
        raise NotImplementedError
    else:
        raise ValueError(
            'non-valid neighborhood type for stencil creation')
    return stencil


def scatter(bounds, count):
    """[summary]

    Arguments:
        bounds {[2d array]} -- [array of two vectors, indicating the bounding box of the scattering envelope with a minimum and maximum of the bounding box]
        count {[int]} -- [number of the points to scatter within the bounding box]

    Returns:
        [cloud] -- [returns a cloud object countaing the coordinates of the scattered points]
    """
    point_array = np.random.uniform(
        bounds[0], bounds[1], (count, bounds.shape[1]))
    return cloud(point_array)


def cloud_from_csv(file_path, delimiter=','):

    point_array = np.genfromtxt(file_path, delimiter=delimiter)
    return cloud(point_array)


def lattice_from_csv(file_path):

    # read the voxel 3-dimensional indices
    ind_flat = np.genfromtxt(file_path, delimiter=',',
                             skip_header=8, usecols=(0, 1, 2)).astype(int)

    # read the voxel values
    vol_flat = np.genfromtxt(
        file_path, delimiter=',', skip_header=8, usecols=(3)).astype(int)

    # read volume meta data
    meta = np.genfromtxt(
        file_path, delimiter='-', skip_header=1, max_rows=3, usecols=(1, 2, 3))
    unit = meta[0]
    min_bound = meta[1]
    volume_shape = meta[2].astype(int)
    max_bound = min_bound + unit * volume_shape

    # reshape the 1d array to get 3d array
    vol = vol_flat.reshape(volume_shape)

    # initializing the lattice
    l = lattice([min_bound, max_bound], unit=unit,
                dtype=bool, default_value=False)

    # setting the latice equal to volume
    l[ind_flat[:, 0], ind_flat[:, 1], ind_flat[:, 2]
      ] = vol[ind_flat[:, 0], ind_flat[:, 1], ind_flat[:, 2]]

    return l


def find_neighbours(lattice, stencil):

    # flatten the lattice
    lattice_flat = lattice.ravel()

    # the id of voxels (0,1,2, ... n)
    lattice_inds = np.arange(lattice.size).reshape(lattice.shape)

    # removing the indecies that are not filled in the volume
    lattice_inds = ((lattice_inds + 1) * lattice) - 1

    # offset the 1-dimensional indices of the voxels that is rshaped to volume shape with value -1
    lattice_inds_paded = np.pad(lattice_inds, (1, 1), mode='constant',
                                constant_values=(-1, -1))

    # flatten
    lattice_inds_paded_flat = lattice_inds_paded.ravel()

    # index of padded cells in flatten
    origin_flat_ind = np.argwhere(lattice_inds_paded_flat != -1).ravel()

    # retrievig all the possible shifts corresponding to the neighbours defined in stencil
    shifts = expand_stencil(stencil)

    # gattering all the replacements in the collumns
    replaced_columns = [
        np.roll(lattice_inds_paded, shift, np.arange(3)).ravel() for shift in shifts]

    # stacking the columns and removing the pads (and also removing the neighbours of the empty voxels since we have tagged them -1 like paddings)
    cell_neighbors = np.stack(replaced_columns, axis=-1)[origin_flat_ind]

    return cell_neighbors


def mesh_sampling(geo_mesh, unit, tol=1e-06, **kwargs):
    """[summary]

    Args:
        geo_mesh ([COMPAS Mesh]): [description]
        unit ([numpy array]): [description]
        tol ([type], optional): [description]. Defaults to 1e-06.

    Returns:
        [type]: [description]
    """
    ####################################################
    # INPUTS
    ####################################################

    dim_num = unit.size
    multi_core_process = kwargs.get('multi_core_process', False)
    return_points = kwargs.get('return_points', False)

    # compare voxel size and tolerance and warn if it is not enough
    if min(unit) * 1e-06 < tol:
        warnings.warn(
            "Warning! The tolerance for rasterization is not small enough, it may result in faulty results or failure of rasterization. Try decreasing the tolerance or scaling the geometry.")

    ####################################################
    # Initialize the volumetric array
    ####################################################

    # retrieve the bounding box information
    mesh_bb = np.array(geo_mesh.bounding_box())
    mesh_bb_min = np.amin(mesh_bb, axis=0)
    mesh_bb_max = np.amax(mesh_bb, axis=0)
    mesh_bb_size = mesh_bb_max - mesh_bb_min

    # find the minimum index in discrete space
    mesh_bb_min_z3 = np.rint(mesh_bb_min / unit).astype(int)
    # calculate the size of voxelated volume
    vol_size = np.ceil((mesh_bb_size / unit)+1).astype(int)
    # initiate the 3d array of voxel space called volume
    vol = np.zeros(vol_size)

    ####################################################
    # claculate the origin and direction of rays
    ####################################################

    # increasing the vol_size by one to accomodate for shooting from corners
    vol_size_off = vol_size + 1
    # retriev the voxel index for ray origins
    hit_vol_ind = np.indices(vol_size_off)
    vol_ind_trans = np.transpose(hit_vol_ind) + mesh_bb_min_z3
    hit_vol_ind = np.transpose(vol_ind_trans)

    # retieve the ray origin indicies
    ray_orig_ind = [np.take(hit_vol_ind, 0, axis=d + 1).transpose((1, 2, 0)).reshape(-1, 3) for d in range(dim_num)]
    ray_orig_ind = np.vstack(ray_orig_ind)

    # retrieve the direction of ray shooting for each origin point
    normals = np.identity(dim_num).astype(int)
    # tile(stamp) the X-ray direction with the (Y-direction * Z-direction) . Then repeat this for all dimensions
    ray_dir = [np.tile(normals[d], (vol_size[(d+1)%dim_num]*vol_size[(d+2)%dim_num], 1)) for d in range(dim_num)]  # this line has a problem given the negative indicies are included now
    ray_dir = np.vstack(ray_dir)

    ####################################################
    # intersection
    ####################################################

    hit_positions = []

    # check if multiprocessing is allowed
    if multi_core_process:
        # open the context manager
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # submit the processes
            results = [executor.submit(
                tri_intersect, geo_mesh, face, unit, mesh_bb_size, ray_orig_ind, ray_dir, tol) for face in geo_mesh.faces()]
            # fetch the results
            for f in concurrent.futures.as_completed(results):
                hit_positions.extend(f.result())
    else:
        # iterate over the faces
        for face in geo_mesh.faces():
            face_hit_pos = tri_intersect(geo_mesh, face, unit, mesh_bb_size,
                                            ray_orig_ind, ray_dir, tol)
            hit_positions.extend(face_hit_pos)

    ####################################################
    # convert hit positions into volumetric data
    ####################################################

    # round the positions to find the closest voxel
    hit_positions = np.array(hit_positions)

    # R3 to Z3
    hit_indicies = np.rint(hit_positions / unit)

    # remove repeated points
    hit_unq_ind = np.unique(hit_indicies, axis=0)

    # calculate volum indecies
    hit_vol_ind = np.transpose(hit_unq_ind - mesh_bb_min_z3).astype(int)

    ####################################################
    # OUTPUTS
    ####################################################

    # Z3 to R3
    # hit_unq_pos = (hit_unq_ind - mesh_bb_min_z3) * unit + mesh_bb_min
    hit_unq_pos = hit_unq_ind * unit

    # set values in the volumetric data
    vol[hit_vol_ind[0], hit_vol_ind[1], hit_vol_ind[2]] = 1

    if return_points:
        return (vol, hit_unq_pos, hit_positions)  # return (vol, hit_unq_pos)
    else:
        return vol

def tri_intersect(geo_mesh, face, unit, mesh_bb_size, ray_orig_ind, ray_dir, tol):
    face_hit_pos = []
    face_verticies_xyz = geo_mesh.face_coordinates(face)
    
    if len(face_verticies_xyz) != 3:
        return([])

    face_verticies_xyz = np.array(face_verticies_xyz)

    # project the ray origin
    proj_ray_orig = ray_orig_ind * unit * (1 - ray_dir)

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

        # calc ray origin position: Z3 to R3
        orig_pos = ray_orig_ind[ray] * unit
        # retrieve ray direction
        direction = ray_dir[ray]
        # calc the destination of ray (max distance that it needs to travel)
        # this line has a problem given the negative indicies are included now
        dest_pos = orig_pos + ray_dir[ray] * mesh_bb_size

        # intersction
        hit_pt = compas.geometry.intersection_line_triangle(
            (orig_pos, dest_pos), face_verticies_xyz, tol=tol)
        if hit_pt is not None:
            face_hit_pos.append(hit_pt)
    
    return(face_hit_pos)