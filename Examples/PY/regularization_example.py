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
reg_pnt_cld = vp.regularization(point_cloud, voxel_size)
# print(reg_pnt_cld)
# print(np.min(point_cloud, axis=0))
# print(np.max(point_cloud, axis=0))

####################################################
# Visualization : PyVista
####################################################

# initiating the plotter
p = pv.Plotter()
p.set_background("black")

# adding the original point cloud: blue
p.add_mesh(pv.PolyData(point_cloud), color='#2499ff',
           point_size=12, render_points_as_spheres=True, label="Original Point Cloud")

# adding the voxel centeroids: red
p.add_mesh(pv.PolyData(reg_pnt_cld), color='#ff244c',
           point_size=15, render_points_as_spheres=True, label="Regularized Point Cloud")

# adding the legend
p.add_legend(bcolor=[0.1, 0.1, 0.1], border=True, size=[0.1, 0.1])

# plotting
p.show()
