
"""
topoGenesis DataStructure
"""
from scipy.spatial.transform import Rotation as sp_rotation
import numpy as np
import pandas as pd
import pyvista as pv
import itertools
import warnings
import os


file_directory = os.path.dirname(os.path.abspath(__file__))


class lattice(np.ndarray):
    """Lattice is subclass of NumPy ndarrays that is adapted to represent 
    a numerical field within a discrete 3dimensional space. It adds spatial 
    properties and functionalities to ndimensional arrays such as bounds, 
    unit, neighbourhood assessment and more.
    """

    def __new__(subtype, bounds, unit=1, dtype=float, buffer=None, offset=0,
                strides=None, order=None, default_value=None, orient=np.array([0., 0., 0., 1.])):

        # TODO: Add documentation for orient
        # extracting min and max from bound and discrtizing it
        bounds = np.array(bounds)
        minbound = np.rint(bounds[0] / unit).astype(int)
        maxbound = np.rint(bounds[1] / unit).astype(int)
        bounds = np.array([minbound, maxbound])*unit

        # unit np array
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

        # set default value
        if default_value != None:
            buffer = np.tile(default_value, shape.flatten())

        # Create the ndarray instance of our type
        obj = super(lattice, subtype).__new__(subtype, shape, dtype,
                                              buffer, offset, strides,
                                              order)

        # set the  'bounds' attribute
        obj.bounds = bounds
        # set the attribute 'unit' to itself
        obj.unit = unit
        # set the 'orient' attribute
        obj.orient = np.array(orient)

        # init an empty connectivity
        obj.connectivity = None
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):

        if obj is None:
            return

        # adding the attributes to self
        self.bounds = getattr(obj, 'bounds', None)
        self.bounds = getattr(obj, 'bounds', None)
        self.unit = getattr(obj, 'unit', None)
        self.orient = getattr(obj, 'orient', None)
        self.connectivity = getattr(obj, 'connectivity', None)
        # TODO need to add the origin atribute

    @property
    def minbound(self):
        """Real minimum bound of the lattice

        Returns:
            numpy.ndarray: real minimum bound
        """
        return self.bounds[0]

    @property
    def maxbound(self):
        """Real maximum bound of the lattice

        Returns:
            numpy.ndarray: real maximum bound
        """
        return self.bounds[1]

    @property
    def centroids(self, threshold=0.0):
        """Extracts the centroid of cells that have a positive or True value and returns them as a point cloud

        Args:
            threshold (float, optional): exclusive lower bound of the values to assume existence of the voxel. Defaults to 0.0.

        Returns:
            topogenesis.Cloud: a point cloud representing the centroids of the lattice cells
        """

        # extract the indices of the True values # with sparse matrix we don't need to search
        point_array = np.argwhere(self > threshold)
        # convert to float
        point_array = point_array.astype(float)
        # scale by unit
        point_array *= self.unit
        # translate by minimum
        point_array += self.minbound
        # orient the points
        if np.sum(np.abs(self.orient[:3])) > 0.001 or self.orient[3] != 1: # check if the orientation is required
            r = sp_rotation.from_quat(self.orient)
            point_array = r.apply(point_array)
        # return as a point cloud
        return cloud(point_array, dtype=float)

    def centroids_threshold(self, threshold=0.0):
        """Extracts the centroid of cells that have a positive or True value and returns them as a point cloud

        Args:
            threshold (float, optional): exclusive lower bound of the values to assume existence of the voxel. Defaults to 0.0.

        Returns:
            topogenesis.Cloud: a point cloud representing the centroids of the lattice cells
        """

        # extract the indices of the True values # with sparse matrix we don't need to search
        point_array = np.argwhere(self > threshold)
        # convert to float
        point_array = point_array.astype(float)
        # scale by unit
        point_array *= self.unit
        # translate by minimum
        point_array += self.minbound
        # orient the points
        if np.sum(np.abs(self.orient[:3])) > 0.001 or self.orient[3] != 1: # check if the orientation is required
            r = sp_rotation.from_quat(self.orient)
            point_array = r.apply(point_array)
        # return as a point cloud
        return cloud(point_array, dtype=float)

    @property
    def indices(self):
        """Creates one-dimensional integer indices for cells in the lattice

        Returns:
            topogenesis.Lattice: integer lattice of indices 
        """

        ind = np.arange(self.size).reshape(self.shape)
        return to_lattice(ind.astype(int), self)

    def fast_vis(self, plot, show_outline: bool = True, show_centroids: bool = True, color = "#ff8fa3", opacity=0.3):
        """Adds the basic lattice features to a pyvista plotter and returns it. 
        It is mainly used to rapidly visualize the content of the lattice 
        for visual confirmation

        Args:
            plot (pyvista.Plotter): a pyvista plotter
            show_outline (bool, optional): If `True`, adds the bounding box of the lattice to the plot
            show_centroids (bool, optional): If `True`, adds the centroid of cells to the plot

        Returns:
            pyvista.Plotter: the same pyvista plotter containing lattice features

        ** Usage Example: **
        ```python
        p = pyvista.Plotter()
        lattice.fast_vis(p)
        ```
        """
        # TODO: Add documentation for color and opacity

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

        # applying the orientation of the lattice
        if np.sum(np.abs(self.orient[:3])) > 0.001 or self.orient[3] != 1: # check if the orientation is required
            Rz = sp_rotation.from_quat(self.orient)
            threshed.points = Rz.apply(threshed.points)

        # adding the voxels: light red
        plot.add_mesh(threshed, show_edges=True, color=color,
                      opacity=opacity, label="Cells")

        if show_outline:
            # adding the boundingbox wireframe
            wireframe = grid.outline()
            Rz = sp_rotation.from_quat(self.orient)
            wireframe.points = Rz.apply(wireframe.points)
            plot.add_mesh(wireframe, color="grey", label="Domain")

        if show_centroids:
            # adding the voxel centroids: red
            plot.add_mesh(pv.PolyData(self.centroids), color='#ff244c', point_size=5,
                          render_points_as_spheres=True, label="Cell Centroids")

        return plot

    def fast_volumetric_vis(self, plot, show_outline: bool = True, show_centroids: bool = True, cmap="coolwarm", clim=[0.5, 1.0], opacity=np.array([0,0.9,0.9,0.9,0.9,0.9,0.9]), value_tag="Value"):
        # TODO: resolve the orientation of the volumetric visualization
        # TODO: Add doc strings

        # Create the spatial reference
        grid = pv.UniformGrid()

        # Set the grid dimensions: shape because we want to inject our values
        grid.dimensions = self.shape
        # The bottom left corner of the data set
        grid.origin = self.minbound
        # These are the cell sizes along each axis
        grid.spacing = self.unit

        # Add the data values to the cell data
        grid.point_arrays[value_tag] = self.flatten(order="F")  # Flatten the Lattice

            
        # adding the volume
        plot.add_volume(grid, cmap=cmap, clim=clim,opacity=opacity, shade=True)

        # # Set the grid dimensions: shape + 1 because we want to inject our values on the CELL data
        # grid = pv.UniformGrid()
        # grid.dimensions = np.array(self.shape) + 1
        # # The bottom left corner of the data set
        # grid.origin = self.minbound - self.unit * 0.5
        # grid.spacing = self.unit  # These are the cell sizes along each axis
        # # Add the data values to the cell data
        # grid.cell_arrays["values"] = self.flatten(
        #     order="F").astype(float)  # Flatten the array!
        # # filtering the voxels
        # threshed = grid.threshold([0.9, 1.1])

        # # applying the orientation of the lattice
        # if np.sum(np.abs(self.orient[:3])) > 0.001 or self.orient[3] != 1: # check if the orientation is required
        #     Rz = sp_rotation.from_quat(self.orient)
        #     threshed.points = Rz.apply(threshed.points)

        # # adding the voxels: light red
        # plot.add_mesh(threshed, show_edges=True, color=color,
        #               opacity=opacity, label="Cells")

        # if show_outline:
        #     # adding the boundingbox wireframe
        #     wireframe = grid.outline()
        #     Rz = sp_rotation.from_quat(self.orient)
        #     wireframe.points = Rz.apply(wireframe.points)
        #     plot.add_mesh(wireframe, color="grey", label="Domain")

        # if show_centroids:
        #     # adding the voxel centroids: red
        #     plot.add_mesh(pv.PolyData(self.centroids), color='#ff244c', point_size=5,
        #                   render_points_as_spheres=True, label="Cell Centroids")

        return plot

    def fast_notebook_vis(self, plot, show_outline: bool = True, show_centroids: bool = True):
        """Adds the basic lattice features to a pyvista ITK plotter and returns it. 
        ITK plotters are specifically used in notebooks to plot the geometry inside 
        the notebook environment It is mainly used to rapidly visualize the content 
        of the lattice for visual confirmation

        Args:
            plot (pyvista.PlotterITK): a pyvista ITK plotter
            show_outline (bool, optional): If `True`, adds the bounding box of the lattice to the plot
            show_centroids (bool, optional): If `True`, adds the centroid of cells to the plot

        Returns:
            pyvista.PlotterITK: pyvista ITK plotter containing lattice features such as cells, bounding box, cell centroids

        ** Usage Example: **
        ```python
        p = pyvista.PlotterITK()
        lattice.fast_notebook_vis(p)    
        ```
        """

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
            # adding the voxel centroids: red
            plot.add_points(pv.PolyData(self.centroids), color='#ff244c')
            # plot.add_mesh(pv.PolyData(self.centroids), color='#ff244c', point_size=5, render_points_as_spheres=True, label="Cell Centroids")

        return plot

    def boolean_marching_cubes(self):
        """This is a polygonization method. It converts the lattice to a boolean lattice and runs a boolean marching cube on the lattice. 

        Returns:
            topogenesis.Lattice: an integer lattice that contains the tile-id at each cell
        """

        # construct the boolean_marching_cubes stencil
        mc_stencil = create_stencil("boolean_marching_cube", 1)

        # retrieve the value of the neighbours
        cell_corners = self.find_neighbours(mc_stencil, order="F")

        # converting volume value (TODO: this needs to become a method of its own)
        volume_flat = self.ravel()
        volume_flat[volume_flat > 0.0] = 1
        volume_flat[volume_flat <= 0.0] = 0

        # replace neighbours by their value in volume
        neighbor_values = volume_flat[cell_corners]

        # computing the cell tile id
        # the powers of 2 in an array
        legend = 2**np.arange(8)

        # multiply the corner with the power of two, sum them, and reshape to the original volume shape
        tile_id = np.sum(legend * neighbor_values,
                         axis=1).reshape(self.shape)

        # drop the last column, row and page (since cube-grid is 1 less than the voxel grid in every dimension)
        # TODO consider that by implementing the origin attribute in lattice this may have to change
        cube_grid = tile_id[:-1, :-1, :-1]

        # convert the array to lattice
        cube_lattice = to_lattice(
            cube_grid, minbound=self.minbound, unit=self.unit)

        return cube_lattice

    def find_connectivity(self, stencil):
        raise NotImplementedError

    def to_csv(self, filepath: str):
        """This method saves the lattice to a csv file

        Args:
            filepath: path to the csv file
        """
        # volume to panda dataframe
        vol_df = self.to_pandas()

        # specifying metadata and transposing it
        metadata = pd.DataFrame({
            'minbound': self.minbound,
            'shape': np.array(self.shape),
            'unit': self.unit,
        })

        with open(filepath, 'w') as df_out:

            metadata.to_csv(df_out, index=False,
                            header=True, float_format='%g', line_terminator='\n')

            df_out.write('\n')

            vol_df.to_csv(df_out,
                          index=False,
                          float_format='%g', line_terminator='\n')

    def to_pandas(self):
        """This methods returns a pandas dataframe containing the lattice information with integer indices and value of the cell as columns.

        Returns:
            pandas.Dataframe: lattice represented in pandas dataframe
        """
        # get the indices of the voxels
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

    # TODO change the default padding value to np.nan. current problem is with datatypes other than float
    def apply_stencil(self, stencil, border_condition: str = "pad_outside", padding_value: int = 0):
        """This method applies the function of a given stencil on the lattice and returns the result in a new lattice.

        Args:
            stencil (topogenesis.Stencil): 
                the stencil to be applied on the lattice
            border_condition (str, optional): 
                specifies how the border condition should be treated. The options are {"pad_outside", "pad_inside", "roll"}. "pad_outside" will offset the lattice in every direction by one step, and fill the new cells with the given `padding_value` and procedes to performing the computation; the resultant lattice in this case has the same shape as the initial lattice. "pad_inside" will perform the computation on the lattice, offsets inside by one cell from each side and returns the remainder cells; the resultant lattice is 2 cell smaller in each dimension than the original lattice. "roll" will assume that the end of each dimension is connected to the beginning of it and interprets the connectivity of the lattice with a rolling approach; the resultant lattice has the same shape is the original lattice. defaults to "pad_outside".
            padding_value (int, optional): 
                value used for padding in case the `border_condition` is set to "pad_outside".

        Raises:
            NotImplementedError: "pad_inside" is not implemented yet

        Returns:
            topogenesis.Lattice: a new lattice containing the result of the application of the stencil
        """

        if border_condition == "pad_outside":
            # pad the volume with zero in every direction
            padded_arr = np.pad(self, (1, 1),
                                mode='constant',
                                constant_values=(padding_value, padding_value))
            # convert to lattice
            self_padded = to_lattice(padded_arr,
                                     self.minbound - self.unit,
                                     unit=self.unit)

        elif border_condition == "pad_inside":
            raise NotImplementedError

        elif border_condition == "roll":
            self_padded = to_lattice(np.copy(self), self)

        # find the neighbours based on the stencil
        cell_neighbors = self_padded.find_neighbours(stencil)

        # replace neighbours by their value in volume
        self_padded_flat = self_padded.ravel()
        neighbor_values = self_padded_flat[cell_neighbors]

        # apply the function to the neighbour values
        applied = stencil.function(neighbor_values, axis=1)

        # reshape the neighbour applied into the original lattice shape
        applied_3d = applied.reshape(self_padded.shape)

        # reverse the padding procedure
        if border_condition == "pad_outside":
            # trim the padded dimensions
            applied_3d_trimmed = applied_3d[1:-1, 1:-1, 1:-1]

        elif border_condition == "pad_inside":
            raise NotImplementedError

        elif border_condition == "roll":
            applied_3d_trimmed = applied_3d

        return to_lattice(applied_3d_trimmed, self)

    def arg_apply_stencil(self, arg_lattice, stencil, border_condition: str = "pad_outside", padding_value: int = 0):
        """Applies the function (should be argument function) of the stencil on the original lattice, extracts the value of the same cell of the argument lattice, fills a new lattice and returns it. If the argument lattice contains one-dimensional ordering of the original lattice, this would function as an argument function. 

        Args:
            arg_lattice (topogenesis.Lattice): 
                the argument lattice. The values in this lattice will be extracted by the argument function and used to fill the new lattice
            stencil (topogenesis.Stencil): 
                the stencil to be applied on the lattice. This stencil should contain an "argument function"
            border_condition (str, optional): 
                specifies how the border condition should be treated. The options are {"pad_outside", "pad_inside", "roll"}. "pad_outside" will offset the lattice in every direction by one step, and fill the new cells with the given `padding_value` and procedes to performing the computation; the resultant lattice in this case has the same shape as the initial lattice. "pad_inside" will perform the computation on the lattice, offsets inside by one cell from each side and returns the remainder cells; the resultant lattice is 2 cell smaller in each dimension than the original lattice. "roll" will assume that the end of each dimension is connected to the beginning of it and interprets the connectivity of the lattice with a rolling approach; the resultant lattice has the same shape is the original lattice. defaults to "pad_outside"
            padding_value (int, optional): 
                value used for padding in case the `border_condition` is set to "pad_outside"

        Raises:
            ValueError: 
                Main lattice and argument lattice shape should match
            NotImplementedError: 
                "pad_inside" is not implemented yet

        Returns:
            topogenesis.Lattice: 
                a new lattice containing the result of the application of the stencil
        """
        if self.shape != arg_lattice.shape:
            raise ValueError(
                "Main lattice and argument lattice shape does not match")

        if border_condition == "pad_outside":
            # pad the volume with padding value in every direction
            padded_arr = np.pad(self, (1, 1),
                                mode='constant',
                                constant_values=(padding_value, padding_value))
            # convert to lattice
            self_padded = to_lattice(padded_arr,
                                     self.minbound - self.unit,
                                     unit=self.unit)
            # pad the argument lattice with padding value in every direction
            padded_arg_arr = np.pad(arg_lattice, (1, 1),
                                    mode='constant',
                                    constant_values=(padding_value, padding_value))
            # convert to lattice
            arg_lattice_padded = to_lattice(padded_arg_arr,
                                            self.minbound - self.unit,
                                            unit=self.unit)

        elif border_condition == "pad_inside":
            raise NotImplementedError

        elif border_condition == "roll":
            self_padded = to_lattice(np.copy(self), self)
            arg_lattice_padded = to_lattice(np.copy(arg_lattice), arg_lattice)

        # find the neighbours based on the stencil
        cell_neighbors = self_padded.find_neighbours(stencil)

        # replace neighbours by their value in the main lattice
        self_padded_flat = self_padded.ravel()
        neighbor_values = self_padded_flat[cell_neighbors]

        # apply the function to the neighbour values
        applied = stencil.function(neighbor_values, axis=1)
        row_ind = np.arange(applied.size)

        # replace neighbours by their value in the argument latice
        arg_lattice_padded_flat = arg_lattice_padded.ravel()
        arg_neighbor_values = arg_lattice_padded_flat[cell_neighbors]

        # retrieve the values from the argument lattice
        arg_applied = arg_neighbor_values[row_ind, applied]

        # reshape the neighbour applied into the original lattice shape
        arg_applied_3d = arg_applied.reshape(self_padded.shape)

        # reverse the padding procedure
        if border_condition == "pad_outside":
            # trim the padded dimensions
            arg_applied_3d_trimmed = arg_applied_3d[1:-1, 1:-1, 1:-1]

        elif border_condition == "pad_inside":
            raise NotImplementedError

        elif border_condition == "roll":
            arg_applied_3d_trimmed = arg_applied_3d

        return to_lattice(arg_applied_3d_trimmed, self)

    def find_neighbours(self, stencil, order: str = "dist"):
        """Given an stencil, this method will return the neighbours of all cells with regard to the stencil specification in a numpy 2D array with each row corresponding to one cell in lattice.

        Args:
            stencil (topogenesis.Stencil): Stencil that describes the neighbourhood
            order (str, optional): the order of neighbours is one of {"dist", "C", "F"}. 'dist' sorts the neighbours based on the distance from origin cell, ‘C’ sorts in row-major (C-style) order, and ‘F’ sorts in column-major (Fortran- style) order. defaults to "dist"

        Returns:
            numpy.ndarray:  2D array describing the neighbours of each cell in the row
        """
        # TODO: kwarg for returning 1d or 3d indices
        # the id of voxels (0,1,2, ... n)
        self_ind = self.indices

        # calculating all the possible shifts to apply to the array
        shifts = stencil.expand(order)

        # gathering all the replacements in the columns
        replaced_columns = [
            np.roll(self_ind, shift, np.arange(3)).ravel() for shift in shifts]

        # stacking the columns
        cell_neighbors = np.stack(replaced_columns, axis=-1)

        return cell_neighbors

    def find_neighbours_masked(self, stencil, loc, order="dist", mask=None, border_condition="standard", id_type="1D"):
        """Given an stencil, this method will return the neighbours of one specific cell (specified with loc) with regard to the neighbourhood that stencil defines. The neighbours can be returned with 1D or 3D indices

        Args:
            stencil (topogenesis.Stencil): Stencil that describes the neighbourhood
            loc (numpy.ndarray): The location (3D index) of the cells that it's neighbours are desired.
            order (str, optional): the order of neighbours is one of {"dist", "C", "F"}. 'dist' sorts the neighbours based on the distance from origin cell, ‘C’ sorts in row-major (C-style) order, and ‘F’ sorts in column-major (Fortran- style) order. defaults to "dist"
            mask (numpy.ndarray, optional): Not implemented yet. Defaults to None.
            border_condition (str, optional): 
                specifies how the border condition should be treated. The options are {"standard", "roll"}. "standard" will assume that the cells at the border of bound of the lattice and they have less neighbours compared to the cells in the middle of the lattice. "roll" will assume that the end of each dimension is connected to the beginning of it and interprets the connectivity of the lattice with a rolling approach; the resultant lattice has the same shape is the original lattice. defaults to "standard"
            id_type (str, optional): specifies how the neighbours are specified and returened. The options are {"1D", "3D"}. "1D" will return a one-dimensional index of the cell, similar to the index that is assigned to each cell when lattice.indices is called. "3D" will return the three-dimensional index of the cell which is the integer coordinates of the cell as well. Defaults to "1D".

        Raises:
            NotImplementedError: in case that "roll" option is chosen for "border_condition"
            NotImplementedError: in case that "mask" option is used

        Returns:
            numpy.ndarray: array of the indices of the neighbours.
        """
        # the id of voxels (0,1,2, ... n)
        self_ind = self.indices

        # find the bounds of window around the specified location
        win_min = loc + stencil.minbound
        win_max = loc + stencil.maxbound + 1
        if border_condition == "standard":
            # ensure win_min is not less than zero
            win_min = np.maximum(win_min, np.array([0, 0, 0]))
            # ensure win_max is not more than shape -1
            win_max = np.minimum(win_max, np.array(self_ind.shape))

        # TODO
        if border_condition == "roll":
            raise NotImplementedError

        # TODO
        if mask != None:
            raise NotImplementedError

        self_ind = self_ind[win_min[0]: win_max[0],
                            win_min[1]: win_max[1],
                            win_min[2]: win_max[2]]

        # TODO: Sometimes, self_ind turns to be zero. need to be debugged
        # find the new 1D-index of the location
        new_loc_ind = np.ravel_multi_index(
            tuple(stencil.origin), self_ind.shape)

        # calculating all the possible shifts to apply to the array
        shifts = stencil.expand(order)

        # gathering all the replacements in the columns
        replaced_columns = [
            np.roll(self_ind, shift, np.arange(3)).ravel() for shift in shifts]

        # stacking the columns
        cell_neighbors = np.stack(replaced_columns, axis=-1)

        # extract 1D ids
        neighs_1d_id = cell_neighbors[new_loc_ind]

        if id_type == "1D":
            return neighs_1d_id

        elif id_type == "3D":
            # convert 1D index to 3D index
            neigh_3d_id = np.array(
                np.unravel_index(neighs_1d_id, self.shape)).T
            return neigh_3d_id


class cloud(np.ndarray):
    """ Cloud is a subclass of NumPy ndarrays to represent pointclouds in a continuous 
    3dimensional space. Clouds add spatial properties and functionalities such
    as bounds and voxelate]
    """

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
        """Real minimum bound of the point cloud

        Returns:
            numpy.ndarray: real minimum bound
        """
        return self.bounds[0]

    @property
    def maxbound(self):
        """Real maximum bound of the point cloud

        Returns:
            numpy.ndarray: real maximum bound
        """
        return self.bounds[1]

    @classmethod
    def from_mesh_vertices(cls, mesh_path: str):
        """ Extracts the vertices of a mesh as the point cloud

        Args:
            mesh_path (str): path to the mesh file

        Returns:
            topogenesis.Cloud: The point cloud including the vertices of the mesh
        """
        # load the mesh using pyvista
        pv_mesh = pv.read(mesh_path)

        # extract vertices
        vertices = np.array(pv_mesh.points).astype(np.float64)

        # return vertices as cloud
        return cls(vertices)

    def voxelate(self, unit, tol=1e-6, **kwargs):
        """will voxelate the pointcloud based on a given unit size and returns a boolean lattice that describes which cells of the discrete space contained at least one point 

        Args:
            unit (int, numpy.array): describes the cell size of the resultant discrete space

        Raises:
            ValueError: if the size of the unit array does not correspond to 
        the number of dimensions in point cloud

        Returns:
            topogenesis.Lattice: boolean lattice describing which cells has contained at least one point
        """

        unit = np.array(unit)
        if unit.size != 1 and unit.size != self.minbound.size:
            raise ValueError(
                'the length of unit array needs to be either 1 or equal to the dimension of point cloud')
        elif unit.size == 1:
            unit = np.tile(unit, self.minbound.shape)

        closed = kwargs.get('closed', False)

        if closed:
            # R3 to Z3 : finding the closest voxel to each point
            point_scaled = self / unit
            # shift the hit points in each 2-dimension (n in 1-axes)
            shifts = np.array(list(itertools.product([-1, 0, 1], repeat=3)))
            shifts_steps = np.sum(np.absolute(shifts), axis=1)
            chosen_shift_ind = np.argwhere(shifts_steps == 3).ravel()
            sel_shifts = shifts[chosen_shift_ind]

            vox_ind = [np.rint(point_scaled + s * tol)
                       for s in sel_shifts]
            vox_ind = np.vstack(vox_ind)

        else:
            vox_ind = np.rint(self / unit).astype(int)

        # removing repetitions
        unique_vox_ind = np.unique(vox_ind, axis=0).astype(int)

        # mapping the voxel indices to real space
        reg_pnt = unique_vox_ind * unit

        # initializing the volume
        l = lattice([self.minbound - unit, self.maxbound + unit], unit=unit,
                    dtype=bool, default_value=False)

        # map the indices to start from zero
        mapped_ind = unique_vox_ind - np.rint(l.minbound/l.unit).astype(int)

        # setting the occupied voxels to True
        l[mapped_ind[:, 0], mapped_ind[:, 1], mapped_ind[:, 2]] = True

        return l

    def fast_vis(self, plot, color="#66beed"):
        """Adds the pointcloud to a pyvista plotter and returns it. 
        It is mainly used to rapidly visualize the point cloud

        Args:
            plot (pyvista.Plotter): a pyvista plotter

        Returns:
            pyvista.Plotter: the same pyvista plotter containing points of the pointcloud

        **Usage Example:**
        ```python
        p = pyvista.Plotter()
        cloud.fast_vis(
        ```
        """

        # adding the original point cloud: blue
        plot.add_mesh(pv.PolyData(self),
                      color=color,
                      point_size=3,
                      render_points_as_spheres=True,
                      label="Point Cloud")

        return plot

    def fast_notebook_vis(self, plot, color="#66beed"):
        """Adds the pointcloud to a pyvista ITK plotter and returns it. ITK plotters are specifically used in notebooks to plot the geometry inside the notebook environment It is mainly used to rapidly visualize the content of the lattice for visual confirmation

        Args:
            plot (pyvista.PlotterITK): a pyvista ITK plotter

        Returns:
            pyvista.PlotterITK: pyvista ITK plotter containing lattice features such as cells, bounding box, cell centroids

        ```python
        p = pyvista.PlotterITK()
        cloud.fast_notebook_vis(p)
        ```
        """
        # adding the original point cloud: blue
        plot.add_points(pv.PolyData(self), color=color)

        return plot

    def to_csv(self, path, delimiter=","):
        np.savetxt(path, self, delimiter=delimiter)


class stencil(np.ndarray):
    """ Stencil is a subclass of NumPy ndarrays to represent neighbourhoods in a discrete 3dimensional space. Certain functions can be assigned to stencil to add functionalities similar to kernels to them. They also provide convenience functions for defining "Moore" or Von Neumann" neighbourhoods. Ufuncs can also be used to alter stencils (Addition, subtraction, etc)]
    """

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

    def expand(self, order: str = "dist"):
        """will return the local address of each filled cell. This can be utilized to access the neighbours of a cell in lattice.

        Args:
            order (str, optional): [description]. Defaults to "dist".

        Returns:
            numpy.ndarray: array in shape of (n, 3) to describe the relative location of neighbours
        """
        # list the locations
        locations = self.origin - np.argwhere(self)

        # check the ordering method

        # 'dist' means to sort based on the distance from origin
        if order == "dist":
            # calculating the distance of each neighbour
            sums = np.abs(locations).sum(axis=1)
            # sorting to identify the main cell
            ordered = np.argsort(sums)

        # ‘F’ means to sort in column-major (Fortran- style) order
        elif order == "F":
            ordered = np.arange(self.size).reshape(self.shape).flatten('F')

        # ‘C’ means to sort in row-major (C-style) order
        elif order == "C":
            ordered = np.arange(self.size).reshape(self.shape).flatten('C')

        # sort and return
        return locations[ordered].astype(int)

    def set_index(self, index, value: int):
        """Sets the value of a cell in stencil via local indexing (based on the origin cell)

        Args:
            index (list, numpy.array): local address of the desired cell
            value (int): the desired value to be set one of {0, 1}

        Raises:
            ValueError: if the local address is non-existence or incompatible with the shape of stencil
        """

        ind = np.array(index) + self.origin
        if ind.size != 3:
            raise ValueError(" the index needs to have three components")
        self[ind[0], ind[1], ind[2]] = value


def create_stencil(type_str: str, steps: int, clip: int = None):
    """Creates a stencil based on predefined neighbourhoods such as "von_neumann" or "moore".

    Args:
        type_str (str): 
            one of {"von_neumann", "moore", "boolean_marching_cube"}
        steps (int): 
            {"von_neumann", "moore"} neighbourhoods are defined based on how many steps far from the origin cell should be included.
        clip (int, optional): 
            will clip the defined neighbourhood, for example ("von_neumann", step=1) describes the 6-neighbourhood of a cell in 3dimensional lattice, ("von_neumann", step=2, clip=1) describes the 18-neighbourhood, and ("moore", step=1) describes 26-neighbourhood, defaults to None

    Raises:
        ValueError: if the neighbourhood type is unknown

    Returns:
        topogenesis.Stencil: the stencil of the perscribed neighbourhood
    """
    # check if clip is specified. if it is not, set it to the steps
    if clip == None:
        clip = steps
    # von neumann neighborhood
    if type_str == "von_neumann":
        # https://en.wikipedia.org/wiki/Von_Neumann_neighborhood

        # computing all the possible shifts to apply to the array
        shifts = np.array(list(itertools.product(
            list(range(-clip, clip+1)), repeat=3)))

        # the number of steps that the neighbour is appart from
        # the cell (step=1 : 6 neighbour, step=2 : 18 neighbours,
        # step=3 : 26 neighbours)
        shift_steps = np.sum(np.absolute(shifts), axis=1)

        # check the number of steps
        chosen_shift_ind = np.argwhere(shift_steps <= steps).ravel()

        # select the valid indices from shifts variable,
        # transpose them to get
        # separate indices in rows, add the number of
        # steps to make this an index
        locs = np.transpose(shifts[chosen_shift_ind]) + clip

        # initialize the stencil
        s = np.zeros((clip*2+1, clip*2+1, clip*2+1)).astype(int)

        # fill in the stencil
        s[locs[0], locs[1], locs[2]] = 1

        return stencil(s,
                       ntype=type_str,
                       origin=np.array([clip, clip, clip]))

    elif type_str == "moore":
        # https://en.wikipedia.org/wiki/Moore_neighborhood

        # computing all the possible shifts to apply to the array
        shifts = np.array(list(itertools.product(
            list(range(-clip, clip+1)), repeat=3)))

        # the number of steps that the neighbour is appart from the origin cell
        shift_steps = np.max(np.absolute(shifts), axis=1)

        # check the number of steps
        chosen_shift_ind = np.argwhere(shift_steps <= steps).ravel()

        # select the valid indices from shifts variable,
        # transpose them to get separate indices in rows,
        # add the number of steps to make this an index
        locs = np.transpose(shifts[chosen_shift_ind]) + clip

        # initialize the stencil
        s = np.zeros((clip*2+1, clip*2+1, clip*2+1)).astype(int)

        # fill in the stencil
        s[locs[0], locs[1], locs[2]] = 1

        return stencil(s, ntype=type_str, origin=np.array([clip, clip, clip]))

    elif type_str == "boolean_marching_cube":

        # initialize the stencil
        s = np.ones((2, 2, 2)).astype(int)

        return stencil(s, ntype=type_str, origin=np.array([0, 0, 0]))

    else:
        raise ValueError(
            'non-valid neighbourhood type for stencil creation')


def to_lattice(a, minbound: np.ndarray, unit=1, orient=np.array([0., 0., 0., 1.])) -> lattice:
    """Converts a numpy array into a lattice

    Args:
        a (numpy.ndarray): 
            array
        minbound (numpy.ndarray): 
            describing the minimum bound of the lattice in continuous space
        unit (int, optional): 
            the unit size of the lattice

    Returns:
        topogenesis.Lattice: the lattice representation of the array
    """
    # check if the minbound is a lattice
    if type(minbound) is lattice:
        l = minbound
        unit = l.unit
        minbound = l.minbound
        orient = l.orient

    # check if minbound is an np array
    elif type(minbound) is not np.ndarray:
        minbound = np.array(minbound)

    # compute the bounds based on the minbound
    bounds = np.array([minbound, minbound + unit * (np.array(a.shape) - 1)])

    # construct a lattice
    array_lattice = lattice(bounds, unit=unit, dtype=a.dtype, orient=orient)
    array_lattice[:, :, :] = a[:, :, :]
    return array_lattice
