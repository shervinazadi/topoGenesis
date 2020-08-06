import os
import numpy as np
import pyvista as pv
import volpy as vp

vs = 0.01
unit = np.array([vs, vs, vs])
tol = 1e-09
mesh_path = os.path.relpath('Examples/SampleData/bunny_lowpoly.obj')
mesh = vp.load_mesh(mesh_path)

sample_cloud = vp.mesh_sampling(mesh, unit, tol=tol)

lattice = sample_cloud.regularize(unit, closed=True)

cube_lattice = lattice.boolean_marching_cubes()

# initiating the plotter
p = pv.Plotter()

p.set_background([0.065, 0.065, 0.065]) # dark grey background
# p.set_background([1.0, 1.0, 1.0])  # white background

# visualize tiles
p = vp.marching_cube_vis(p, cube_lattice, "chamfer")

# # fast visualization of the lattice
lattice.fast_vis(p)

# adding the base mesh: light blue
# mesh = pv.read(geo_path)
# p.add_mesh(mesh, show_edges=True, color='#abd8ff', opacity=0.4, label="Base Mesh")

# Set a camera position
p.camera_position = [(0.25, 0.18, 0.5), (0, .1, 0), (0, 1, 0)]

# plotting
p.show()