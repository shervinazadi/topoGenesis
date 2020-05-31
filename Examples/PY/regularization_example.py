import numpy as np

pointcloud_path = "Examples/SampleData/PointCloud.csv"
pointcloud = np.genfromtxt(pointcloud_path, delimiter=",")

print(np.min(pointcloud, axis=0))
print(np.max(pointcloud, axis=0))
