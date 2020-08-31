
"""
Lattice DataStructure
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
        # convert to float
        point_array = point_array.astype(float)
        # multply by unit
        point_array *= self.unit
        # move to minimum
        point_array += self.minbound
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
        plot.add_mesh(threshed, show_edges=True, color="#ff8fa3", opacity=0.3, label="Cells")

        if show_outline:
            # adding the boundingbox wireframe
            plot.add_mesh(grid.outline(), color="grey", label="Domain")

        if show_centroids:
            # adding the voxel centeroids: red
            plot.add_mesh(pv.PolyData(self.centroids), color='#ff244c', point_size=5, render_points_as_spheres=True, label="Cell Centroidss")

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
        replaced_columns = [np.roll(volume_inds, shift, np.arange(3)).ravel() for shift in shifts]

        # stacking the columns
        cell_corners = np.stack(replaced_columns, axis=-1)

        # converting volume value (TODO: this needs to become a method of its own)
        volume_flat = volume.ravel()
        volume_flat[volume_flat>0.5] = 1
        volume_flat[volume_flat<0.5] = 0

        # replace neighbours by their value in volume
        neighbor_values = volume_flat[cell_corners]

        # computing the cell tile id
        # the powers of 2 in an array
        legend = 2**np.arange(8)

        # multiply the corner with the power of two, sum them, and reshape to the original volume shape
        tile_id = np.sum(legend * neighbor_values, axis=1).reshape(volume.shape)

        # drop the last column, row and page (since cube-grid is 1 less than the voxel grid in every dimension)
        # TODO consider that removing padding would eliminate the need for this line
        cube_grid = tile_id[:-1, :-1, :-1]

        # initializing the lattice
        cube_lattice = lattice([self.minbound, self.maxbound + self.unit], unit=self.unit, dtype=np.uint8, buffer=cube_grid, default_value=False)
        
        # set the values that are bigger than 0 (transfering values)
        cube_lattice[cube_grid>0] = cube_grid[cube_grid>0] 

        return cube_lattice

    def find_connectivity(self, stencil):
        raise NotImplementedError

