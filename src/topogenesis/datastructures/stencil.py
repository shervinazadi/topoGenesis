
"""
Stencil DataStructure
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

class stencil(np.ndarray):

    def __new__(subtype, point_array, ntype="Custom", origin=np.array([0,0,0]), dtype=int, buffer=None, offset=0,
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
        
        if sort=="dist": # Sorted Based on the distance from origin
            # calculating the distance of each neighbour
            sums = np.abs(locations).sum(axis=1)
            # sorting to identify the main cell
            order = np.argsort(sums)

        elif sort=="F": # Fortran Sort, used for Boolean Marching Cubes
            order = np.arange(self.size).reshape(self.shape).flatten('F')

        # sort and return
        return locations[order].astype(int)
    
    def set_index(self, index, value):
        ind = np.array(index) + self.origin
        if ind.size != 3:
            raise ValueError(" the index needs to have three components")
        self[ind[0],ind[1],ind[2]] = value


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

        # inilize the stencil
        s = np.zeros((clip*2+1, clip*2+1, clip*2+1)).astype(int)

        # fill in the stencil
        s[locs[0], locs[1], locs[2]] = 1

        return stencil(s, ntype=type_str, origin=np.array([clip, clip, clip]))

    elif type_str == "moore":
        # https://en.wikipedia.org/wiki/Moore_neighborhood
        
        # claculating all the possible shifts to apply to the array
        shifts = np.array(list(itertools.product(
            list(range(-clip, clip+1)), repeat=3)))

        # the number of steps that the neighbour is appart from the origin cell
        shift_steps = np.max(np.absolute(shifts), axis=1)

        # check the number of steps
        chosen_shift_ind = np.argwhere(shift_steps <= steps).ravel()
        
        # select the valid indices from shifts variable, transpose them to get separate indicies in rows, add the number of steps to make this an index
        locs = np.transpose(shifts[chosen_shift_ind]) + clip

        # inilize the stencil
        s = np.zeros((clip*2+1, clip*2+1, clip*2+1)).astype(int)

        # fill in the stencil
        s[locs[0], locs[1], locs[2]] = 1

        return stencil(s, ntype=type_str, origin=np.array([clip, clip, clip]))

    elif type_str == "boolean_marching_cube":
        
        # # shifts to check 8 corner of cube (multiply by -1 since shift goes backward)
        # shifts = np.array([
        #     [0, 0, 0],  # 1
        #     [1, 0, 0],  # 2
        #     [0, 1, 0],  # 4
        #     [1, 1, 0],  # 8
        #     [0, 0, 1],  # 16
        #     [1, 0, 1],  # 32
        #     [0, 1, 1],  # 64
        #     [1, 1, 1]   # 128
        # ])*-1

        # # the number of steps that the neighbour is appart from the origin cell
        # shift_steps = np.max(np.absolute(shifts), axis=1)

        # # check the number of steps
        # chosen_shift_ind = np.argwhere(shift_steps <= steps).ravel()
        
        # # select the valid indices from shifts variable, transpose them to get separate indicies in rows, add the number of steps to make this an index
        # locs = np.transpose(shifts[chosen_shift_ind]) + clip

        # inilize the stencil
        s = np.ones((2,2,2)).astype(int)

        # # fill in the stencil
        # s[locs[0], locs[1], locs[2]] = 1

        return stencil(s, ntype=type_str, origin=np.array([0,0,0]))

    else:
        raise ValueError(
            'non-valid neighborhood type for stencil creation')

