
"""
Rasterizing a mesh to a volumetric datastructure
"""
import os
import numpy as np
import compas.datastructures as ds
import pyvista as pv
import topogenesis as tg

__author__ = "Shervin Azadi, and Pirouz Nourian"
__copyright__ = "???"
__credits__ = ["Shervin Azadi", "Pirouz Nourian"]
__license__ = "???"
__version__ = "0.0.2"
__maintainer__ = "Shervin Azadi"
__email__ = "shervinazadi93@gmail.com"
__status__ = "Dev"

####################################################
# INPUTS
####################################################

vs = 0.01
unit = np.array([vs, vs, vs])
tol = 1e-09
mesh_path = os.path.relpath('data/bunny_lowpoly.obj')
mesh = tg.geometry.load_mesh(mesh_path)

####################################################
# Mesh Sample
####################################################

sample_cloud, ray_origins = tg.geometry.mesh_sampling(
    mesh, unit, multi_core_process=False, return_ray_origin=True, tol=tol)

####################################################
# Voxelation
####################################################

lattice = sample_cloud.voxelate(unit, closed=True)

####################################################
# Visualization : PyVista
####################################################

# initiating the plotter
p = pv.Plotter()

# p.set_background([0.065, 0.065, 0.065]) # dark grey background
p.set_background([1.0, 1.0, 1.0])  # white background

# fast visualization of the point cloud
sample_cloud.fast_vis(p)

# fast visualization of the lattice
lattice.fast_vis(p)

# adding the base mesh: light blue
mesh = pv.read(mesh_path)
p.add_mesh(mesh, show_edges=True, color='#abd8ff',
           opacity=0.4, label="Base Mesh")

# adding the ray origins: dark blue
p.add_mesh(pv.PolyData(ray_origins), color='#004887', point_size=4,
           render_points_as_spheres=True, label="Ray Origins")

# adding the legend
p.add_legend(bcolor=[0.9, 0.9, .9], border=True, size=[0.1, 0.1])

# Set a camera position
p.camera_position = [(0.25, 0.18, 0.5), (0, .1, 0), (0, 1, 0)]

# plotting
p.show()
