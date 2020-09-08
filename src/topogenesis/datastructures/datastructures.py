
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

__author__ = "Shervin Azadi, and Pirouz Nourian"
__copyright__ = "???"
__credits__ = ["Shervin Azadi", "Pirouz Nourian"]
__license__ = "???"
__version__ = "0.0.2"
__maintainer__ = "Shervin Azadi"
__email__ = "shervinazadi93@gmail.com"
__status__ = "Dev"

file_directory = os.path.dirname(os.path.abspath(__file__))


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

        # if the unit vector size is 1, tile it with the size of minimum vector
        if unit.size == 1:
            unit = np.tile(unit, minbound.size)
        # raise value error if the size of unit is neighter 1 nor the length of the minimum
        elif unit.size != minbound.size:
            raise ValueError(
                'the length of unit array needs to be either 1 or equal to the min/max arrays')

        # calculating shape based on bounds and unit
        shape = 1 + maxbound - minbound

        # set defualt value
        if default_value != None:
            buffer = np.tile(default_value, shape.flatten())

        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to lattice.__array_finalize__
        obj = super(lattice, subtype).__new__(subtype, shape, dtype,
                                              buffer, offset, strides,
                                              order)

        # set the  'bounds' attribute
        obj.bounds = bounds
        # set the attribute 'unit' to itself
        obj.unit = unit

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
        # convert to float
        point_array = point_array.astype(float)
        # multply by unit
        point_array *= self.unit
        # move to minimum
        point_array += self.minbound
        # return as a point cloud
        return cloud(point_array, dtype=float)

    @property
    def indicies(self):
        ind = np.arange(self.size).reshape(self.shape)
        return to_lattice(ind.astype(int), self)

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

    def fast_notebook_vis(self, plot, show_outline=True, show_centroids=True):

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
        plot.add_mesh(threshed, color="#ff8fa3", opacity=0.3)
        # plot.add_mesh(threshed, show_edges=True, color="#ff8fa3", opacity=0.3, label="Cells")

        if show_outline:
            # adding the boundingbox wireframe
            plot.add_mesh(grid.outline(), color="grey")
            # plot.add_mesh(grid.outline(), color="grey", label="Domain")

        if show_centroids:
            # adding the voxel centeroids: red
            plot.add_points(pv.PolyData(self.centroids), color='#ff244c')
            # plot.add_mesh(pv.PolyData(self.centroids), color='#ff244c', point_size=5, render_points_as_spheres=True, label="Cell Centroidss")

        return plot

    def boolean_marching_cubes(self):

        # construct the boolean_marching_cubes stencil
        mc_stencil = create_stencil("boolean_marching_cube", 1)

        # getting shifts by expanding the stencil in the Fortran Order
        shifts = mc_stencil.expand('F')

        # pad the volume with zero in every direction
        # TODO make this an option instead of default
        volume = np.pad(self, (1, 1), mode='constant', constant_values=(0, 0))

        # the id of voxels (0,1,2, ... n)
        volume_inds = np.arange(volume.size).reshape(volume.shape)

        # gattering all the replacements in the collumns
        replaced_columns = [
            np.roll(volume_inds, shift, np.arange(3)).ravel() for shift in shifts]

        # stacking the columns
        cell_corners = np.stack(replaced_columns, axis=-1)

        # converting volume value (TODO: this needs to become a method of its own)
        volume_flat = volume.ravel()
        volume_flat[volume_flat > 0.5] = 1
        volume_flat[volume_flat < 0.5] = 0

        # replace neighbours by their value in volume
        neighbor_values = volume_flat[cell_corners]

        # computing the cell tile id
        # the powers of 2 in an array
        legend = 2**np.arange(8)

        # multiply the corner with the power of two, sum them, and reshape to the original volume shape
        tile_id = np.sum(legend * neighbor_values,
                         axis=1).reshape(volume.shape)

        # drop the last column, row and page (since cube-grid is 1 less than the voxel grid in every dimension)
        # TODO consider that removing padding would eliminate the need for this line
        cube_grid = tile_id[:-1, :-1, :-1]

        # initializing the lattice
        cube_lattice = lattice([self.minbound, self.maxbound + self.unit],
                               unit=self.unit, dtype=np.uint8, buffer=cube_grid, default_value=False)

        # set the values that are bigger than 0 (transfering values)
        cube_lattice[cube_grid > 0] = cube_grid[cube_grid > 0]

        return cube_lattice

    def find_connectivity(self, stencil):
        raise NotImplementedError

    def to_csv(self, filepath):
        # volume to panda dataframe
        vol_df = self.to_panadas()

        # specifying metadata and transposig it
        metadata = pd.DataFrame({
            'minbound': self.minbound,
            'shape': np.array(self.shape),
            'unit': self.unit,
        })

        with open(filepath, 'w') as df_out:

            metadata.to_csv(df_out, index=False,
                            header=True, float_format='%g')

            df_out.write('\n')

            vol_df.to_csv(df_out, index=False, float_format='%g')

    def to_panadas(self):
        # get the indicies of the voxels
        vol_3d_ind = np.indices(self.shape)

        # flatten except the last dimension
        vol_3d_ind_flat = vol_3d_ind.transpose(1, 2, 3, 0).reshape(-1, 3)

        # flatten the volume
        vol_flat = self.ravel()

        # volume data to panda dataframe
        vol_df = pd.DataFrame(
            {
                'IX': vol_3d_ind_flat[:, 0],
                'IY': vol_3d_ind_flat[:, 1],
                'IZ': vol_3d_ind_flat[:, 2],
                'value': vol_flat,
            })
        return vol_df
    # TODO change the defualt padding value to np.nan. current problem is with datatypes other than float
    def apply_stencil(self, stencil, border_condition="pad_outside", padding_value=0):

        if border_condition == "pad_outside":
            # pad the volume with zero in every direction
            self_padded = np.pad(self, (1, 1), mode='constant',
                                 constant_values=(padding_value, padding_value))

        elif border_condition == "pad_inside":
            raise NotImplementedError

        elif border_condition == "roll":
            self_padded = np.copy(self)

        # the id of voxels (0,1,2, ... n)
        self_padded_inds = np.arange(
            self_padded.size).reshape(self_padded.shape)

        # claculating all the possible shifts to apply to the array
        shifts = stencil.expand()

        # gattering all the replacements in the collumns
        replaced_columns = [
            np.roll(self_padded_inds, shift, np.arange(3)).ravel() for shift in shifts]

        # stacking the columns
        cell_neighbors = np.stack(replaced_columns, axis=-1)

        # replace neighbours by their value in volume
        self_padded_flat = self_padded.ravel()
        neighbor_values = self_padded_flat[cell_neighbors]

        # apply the function to the neighbour values
        applied = stencil.function(neighbor_values, axis=1)

        # reshape the neighbour applied into the origial lattice shape
        applied_3d = applied.reshape(self_padded.shape)

        # reverse the padding procedure
        if border_condition == "pad_outside":
            # trim the padded dimensions
            applied_3d_trimed = applied_3d[1:-1, 1:-1, 1:-1]

        elif border_condition == "pad_inside":
            raise NotImplementedError

        elif border_condition == "roll":
            applied_3d_trimed = applied_3d

        return to_lattice(applied_3d_trimed, self)


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

    def voxelate(self, unit, **kwargs):
        """[summary]

        Arguments:
            unit {[float or array of floats]} -- [the unit separation between cells of lattice]

        Keyword Arguments:
            closed {[Boolean]} -- [False by default. If the cell intervals are closed intervals or not.]

        Raises:
            ValueError: [unit needs to be either a float or an array of floats that has the same dimension of the points in the point cloud]

        Returns:
            [lattice] -- [a boolian latice representing the rasterization of point cloud]
        """
        ####################################################
        # INPUTS
        ####################################################
        unit = np.array(unit)
        if unit.size != 1 and unit.size != self.minbound.shape:
            raise ValueError(
                'the length of unit array needs to be either 1 or equal to the dimension of point cloud')
        elif unit.size == 1:
            unit = np.tile(unit, self.minbound.shape)

        closed = kwargs.get('closed', False)

        ####################################################
        # PROCEDURE
        ####################################################

        if closed:
            # retrieve the identity matrix as a list of main axes
            axes = np.identity(unit.size).astype(int)
            # R3 to Z3 : finding the closest voxel to each point
            point_scaled = self / unit
            # shift the hit points in each 2-dimension (n in 1-axes) backward and formard (s in [-1,1]) and rint all the possibilities
            vox_ind = [np.rint(point_scaled + unit * n * s * 0.001)
                       for n in (1-axes) for s in [-1, 1]]
            vox_ind = np.vstack(vox_ind)
        else:
            vox_ind = np.rint(self / unit).astype(int)

        # removing repetitions
        unique_vox_ind = np.unique(vox_ind, axis=0).astype(int)

        # mapping the voxel indicies to real space
        reg_pnt = unique_vox_ind * unit

        # initializing the volume
        l = lattice([self.minbound, self.maxbound], unit=unit,
                    dtype=bool, default_value=False)

        # mapp the indicies to start from zero
        mapped_ind = unique_vox_ind - np.rint(l.bounds[0]/l.unit).astype(int)

        # setting the occupied voxels to True
        l[mapped_ind[:, 0], mapped_ind[:, 1], mapped_ind[:, 2]] = True

        ####################################################
        # OUTPUTS
        ####################################################

        return l

    def fast_vis(self, plot):

        # adding the original point cloud: blue
        plot.add_mesh(pv.PolyData(self),
                      color='#2499ff',
                      point_size=3,
                      render_points_as_spheres=True,
                      label="Point Cloud")

        return plot

    def fast_notebook_vis(self, plot):

        # adding the original point cloud: blue
        plot.add_points(pv.PolyData(self), color='#2499ff')

        return plot

    def to_csv(self, path, delimiter=","):
        np.savetxt(path, self, delimiter=delimiter)


class stencil(np.ndarray):

    def __new__(subtype, point_array, ntype="Custom", origin=np.array([0, 0, 0]), function=None, dtype=int, buffer=None, offset=0,
                strides=None, order=None):

        # extracting the shape from point_array
        shape = point_array.shape
        # using the point_array as the buffer
        buffer = point_array.flatten(order="C")

        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to stencil.__array_finalize__
        obj = super(stencil, subtype).__new__(subtype, shape, dtype,
                                              buffer, offset, strides,
                                              order)

        # set the neighbourhood type
        obj.ntype = ntype
        # set the origin
        obj.origin = origin
        # set the  'bounds' attribute
        shape_arr = np.array(shape)
        obj.bounds = np.array([shape_arr * 0, shape_arr - 1]) - origin
        # set the function attribute
        obj.function = function
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(stencil, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. stencil():
        #    obj is None
        #    (we're in the middle of the stencil.__new__
        #    constructor, and self.bounds will be set when we return to
        #    stencil.__new__)
        if obj is None:
            return
        # From view casting - e.g arr.view(stencil):
        #    obj is arr
        #    (type(obj) can be stencil)
        # From new-from-template - e.g stencil[:3]
        #    type(obj) is stencil
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'properties', because this
        # method sees all creation of default objects - with the
        # stencil.__new__ constructor, but also with
        # arr.view(stencil).
        self.bounds = getattr(obj, 'bounds', None)
        self.ntype = getattr(obj, 'ntype', None)
        self.origin = getattr(obj, 'origin', None)
        self.function = getattr(obj, 'function', None)

        # We do not need to return anything

    def __array_wrap__(self, array, context=None):

        temp = np.array(array)

        # checking if the array has any value other than 0, and 1
        np.place(temp, temp > 0.5, [1])
        np.place(temp, temp < 0.5, [0])

        return stencil(temp, ntype="custom", origin=self.origin)

    @property
    def minbound(self):
        return self.bounds[0]

    @property
    def maxbound(self):
        return self.bounds[1]

    def expand(self, sort="dist"):
        # list the locations
        locations = self.origin - np.argwhere(self)

        # check the sorting method

        if sort == "dist":  # Sorted Based on the distance from origin
            # calculating the distance of each neighbour
            sums = np.abs(locations).sum(axis=1)
            # sorting to identify the main cell
            order = np.argsort(sums)

        elif sort == "F":  # Fortran Sort, used for Boolean Marching Cubes
            order = np.arange(self.size).reshape(self.shape).flatten('F')

        # sort and return
        return locations[order].astype(int)

    def set_index(self, index, value):
        ind = np.array(index) + self.origin
        if ind.size != 3:
            raise ValueError(" the index needs to have three components")
        self[ind[0], ind[1], ind[2]] = value


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

        # the number of steps that the neighbour is appart from
        # the cell (setp=1 : 6 neighbour, step=2 : 18 neighbours,
        # step=3 : 26 neighbours)
        shift_steps = np.sum(np.absolute(shifts), axis=1)

        # check the number of steps
        chosen_shift_ind = np.argwhere(shift_steps <= steps).ravel()

        # select the valid indices from shifts variable,
        # transpose them to get
        # separate indicies in rows, add the number of
        # steps to make this an index
        locs = np.transpose(shifts[chosen_shift_ind]) + clip

        # inilize the stencil
        s = np.zeros((clip*2+1, clip*2+1, clip*2+1)).astype(int)

        # fill in the stencil
        s[locs[0], locs[1], locs[2]] = 1

        return stencil(s,
                       ntype=type_str,
                       origin=np.array([clip, clip, clip]))

    elif type_str == "moore":
        # https://en.wikipedia.org/wiki/Moore_neighborhood

        # claculating all the possible shifts to apply to the array
        shifts = np.array(list(itertools.product(
            list(range(-clip, clip+1)), repeat=3)))

        # the number of steps that the neighbour is appart from the origin cell
        shift_steps = np.max(np.absolute(shifts), axis=1)

        # check the number of steps
        chosen_shift_ind = np.argwhere(shift_steps <= steps).ravel()

        # select the valid indices from shifts variable,
        # transpose them to get separate indicies in rows,
        # add the number of steps to make this an index
        locs = np.transpose(shifts[chosen_shift_ind]) + clip

        # inilize the stencil
        s = np.zeros((clip*2+1, clip*2+1, clip*2+1)).astype(int)

        # fill in the stencil
        s[locs[0], locs[1], locs[2]] = 1

        return stencil(s, ntype=type_str, origin=np.array([clip, clip, clip]))

    elif type_str == "boolean_marching_cube":

        # inilize the stencil
        s = np.ones((2, 2, 2)).astype(int)

        return stencil(s, ntype=type_str, origin=np.array([0, 0, 0]))

    else:
        raise ValueError(
            'non-valid neighborhood type for stencil creation')


def to_lattice(a, l):
    # construct a lattice
    array_lattice = lattice(l.bounds, unit=l.unit, dtype=a.dtype)
    array_lattice[:, :, :] = a[:, :, :]
    return array_lattice
