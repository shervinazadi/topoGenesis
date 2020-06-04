import volpy as vp
import pyvista as pv

# read pint cloud from file
point_cloud_path = "Examples/SampleData/PointCloud.csv"
pc = vp.gen_from_csv(point_cloud_path)

# regularizing random points into a lattice
l = pc.regularize([1, 1, 1])

# initiating the pyvista plotter
p = pv.Plotter()
p.set_background([0.065, 0.065, 0.065])

# add the lattice to the visualization
p = l.fast_vis(p, show_outline=True, show_centroids=True)

# add the original point cloud to the visualization
p = pc.fast_vis(p)

# adding the legend
p.add_legend(bcolor=[0.1, 0.1, 0.1], border=True, size=[0.1, 0.1])
# plotting
p.show()
