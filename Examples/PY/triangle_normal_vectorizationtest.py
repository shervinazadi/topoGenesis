import numpy as np
import volpy as vp


def surface_normal_newell(poly):
    # https://stackoverflow.com/questions/39001642/calculating-surface-normal-in-python-using-newells-method
    # Newell Method explained here: https://www.researchgate.net/publication/324921216_Topology_On_Topology_and_Topological_Data_Models_in_Geometric_Modeling_of_Space

    n = np.array([0.0, 0.0, 0.0])

    for i in range(3):
        j = (i+1) % 3
        n[0] += (poly[i][1] - poly[j][1]) * (poly[i][2] + poly[j][2])
        n[1] += (poly[i][2] - poly[j][2]) * (poly[i][0] + poly[j][0])
        n[2] += (poly[i][0] - poly[j][0]) * (poly[i][1] + poly[j][1])

    norm = np.linalg.norm(n)
    if norm == 0:
        raise ValueError('zero norm')
    else:
        normalised = n/norm

    return normalised


tri = np.array([[-0.00601774128, 0.130592465, 0.0237104725],
                [-0.0866273791, 0.153729707, 0.0216472838],
                [-0.0290798154, 0.125226036, 0.00471670832]])

print(surface_normal_newell(tri))
print(vp.surface_normal_newell(tri))
