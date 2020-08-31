
"""
Cloud DataStructure
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

    def regularize(self, unit, **kwargs):
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
        if unit.size != 1 and unit.size != self.bounds.shape[1]:
            raise ValueError(
                'the length of unit array needs to be either 1 or equal to the dimension of point cloud')
        elif unit.size == 1:
            unit = np.tile(unit, (1, self.bounds.shape[1]))

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
            vox_ind = [np.rint(point_scaled + unit * n * s * 0.001) for n in (1-axes) for s in [-1, 1]]
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
        plot.add_mesh(pv.PolyData(self), color='#2499ff', point_size=3, render_points_as_spheres=True, label="Point Cloud")

        return plot

    def fast_notebook_vis(self, plot):

        # adding the original point cloud: blue
        plot.add_points(pv.PolyData(self), color='#2499ff')

        return plot


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
