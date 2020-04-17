

import numpy as np
import pyvista as pv

# ind = np.indices((30,40,50))
# nodes = np.transpose(ind, (1,2,3,0))

# # mesh = pv.PolyData(nodes)
# # mesh.plot(point_size=5)

# vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 0.5, 0], [0, 0.5, 0],])

# faces = np.hstack([[3, 0, 1, 2], [3, 0, 3, 2]]).astype(np.int8)
# surf = pv.PolyData(vertices, faces)
# surf.plot(point_size=1)

mesh = pv.read("Examples/IN/bunny.obj")

p = pv.Plotter()
p.add_mesh(mesh, show_edges=True, color='white', opacity=0.8)
p.add_mesh(pv.PolyData(mesh.points), color='red',
       point_size=5, render_points_as_spheres=True)
p.camera_position = [(0.02, 0.30, 0.73),(0.02, 0.03, -0.022),(-0.03, 0.94, -0.34)]
p.show()