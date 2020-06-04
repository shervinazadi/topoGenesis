import numpy as np
import volpy as vp
import pyvista as pv

# loading point clouds
point_cloud_path = "Examples/SampleData/PointCloud.csv"
point_cloud = np.genfromtxt(point_cloud_path, delimiter=",")

# regularization voxel size
vs = 1
voxel_size = np.array([vs, vs, vs])

# regularization
reg_pnt_cld, vol = vp.regularization(point_cloud, voxel_size, return_vol=True)

####################################################
# Visualization : PyVista
####################################################

# Set the grid dimensions: shape + 1 because we want to inject our values on the CELL data
grid = pv.UniformGrid()
grid.dimensions = np.array(vol.shape) + 1
# The bottom left corner of the data set
grid.origin = np.min(reg_pnt_cld, axis=0) - voxel_size * 0.5
grid.spacing = voxel_size  # These are the cell sizes along each axis
# Add the data values to the cell data
grid.cell_arrays["values"] = vol.flatten(order="F")  # Flatten the array!
# filtering the voxels
threshed = grid.threshold([0.9, 1.1])
# bounding box of the voxelation
outline = grid.outline()

# initiating the plotter
p = pv.Plotter()
p.set_background([0.065, 0.065, 0.065])

# adding the original point cloud: blue
p.add_mesh(pv.PolyData(point_cloud), color='#2499ff',
           point_size=2, render_points_as_spheres=True, label="Original Point Cloud")

# adding the voxel centeroids: red
p.add_mesh(pv.PolyData(reg_pnt_cld), color='#ff244c',
           point_size=5, render_points_as_spheres=True, label="Regularized Point Cloud")

# adding the voxels: light red
p.add_mesh(threshed, show_edges=True, color="#ff8fa3",
           opacity=0.3, label="Voxels")

# adding the boundingbox wireframe
p.add_mesh(outline, color="grey", label="Domain")

# adding the legend
p.add_legend(bcolor=[0.1, 0.1, 0.1], border=True, size=[0.1, 0.1])

# plotting
p.show()
