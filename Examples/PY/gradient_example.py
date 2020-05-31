import volpy as vp
import numpy as np

# set the shape of aray
s = 3

# init the scalar field
X, Y, Z = np.mgrid[-s:s:30j, -s:s:30j, -s:s:30j]
val = np.sin(X * Y * Z) / (X * Y * Z)

grad_val = vp.gradient(val)

print(grad_val)
