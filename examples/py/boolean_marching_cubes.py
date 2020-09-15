import os               # for path manipulation
import topogenesis as tg      # core
import pyvista as pv    # for plotting and visualizations

# Step 0: Specifing all the inputs

vs = 0.01               # voxel size
unit = [vs, vs, vs]     # unit size
tol = 1e-09             # intersection tolerance
mesh_path = os.path.relpath('data/bunny_lowpoly.obj')
original_mesh = tg.geometry.load_mesh(mesh_path)

# Step 1: Sampling the mesh and constructing the point cloud
sample_cloud = tg.geometry.mesh_sampling(original_mesh, unit, tol=tol)

# Step 2: Voxelating the point cloud to construct the lattice
lattice = sample_cloud.voxelate(unit, closed=True)

# Step 3: Constructing the Cube Lattice using the Boolea Marching Cube Algorithm
cube_lattice = lattice.boolean_marching_cubes()

# Step 4: Plotting
# initiating the plotter
p = pv.Plotter()

p.set_background([0.065, 0.065, 0.065])  # dark grey background
# p.set_background([1.0, 1.0, 1.0])  # white background

# visualize tiles
p = tg.marching_cube_vis(p, cube_lattice, "chamfer")

# # fast visualization of the lattice
lattice.fast_vis(p)

# adding the base mesh: light blue
# mesh = pv.read(geo_path)
# p.add_mesh(mesh, show_edges=True, color='#abd8ff', opacity=0.4, label="Base Mesh")

# Set a camera position
p.camera_position = [(0.25, 0.18, 0.5), (0, .1, 0), (0, 1, 0)]

# plotting
p.show()
