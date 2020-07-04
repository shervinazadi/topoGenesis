import numpy as np
import pyvista as pv
import itertools


class lattice(np.ndarray):

    def __new__(subtype, bounds, unit=1, dtype=float, buffer=None, offset=0,
                strides=None, order=None, default_value=None):

        # extracting min and max from bound and discrtizing it
        bounds = np.array(bounds)
        minbound = np.rint(bounds[0] / unit).astype(int)
        maxbound = np.rint(bounds[1] / unit).astype(int)
        bounds = np.array([minbound, maxbound])

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
        self.dis_bounds = getattr(obj, 'dis_bounds', None)
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


class stencil(lattice):
    """[This will be class based on the latice class]

    Args:
        lattice ([type]): [description]

    Returns:
        [type]: [description]
    """
    pass


def expand_stencil(stencil):
    locations = np.argwhere(stencil) - (stencil.shape[0] - 1)/2
    return locations.astype(int)


def create_stencil(type_str, steps, clip=None):
    # check if clip is specified. if it is not, set it to the steps
    if clip == None:
        clip = steps
    # von neumann neighborhood
    if type_str == "von_neumann":
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
