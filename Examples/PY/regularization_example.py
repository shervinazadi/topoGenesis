import numpy as np
import volpy as vp

# loading point clouds
point_cloud_path = "Examples/SampleData/PointCloud.csv"
point_cloud = np.genfromtxt(point_cloud_path, delimiter=",")

# regularization voxel size
vs = 1
voxel_size = np.array([vs, vs, vs])

# regularization
reg_pnt_cld = vp.regularization(point_cloud, voxel_size)
print(reg_pnt_cld)
print(np.min(point_cloud, axis=0))
print(np.max(point_cloud, axis=0))
