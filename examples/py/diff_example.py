import topogenesis as tg
import numpy as np

# set the shape of aray
s = 3

# init the scalar field
X, Y, Z = np.mgrid[-s:s:30j, -s:s:30j, -s:s:30j]
val = np.sin(X * Y * Z) / (X * Y * Z)

# gradient
grad_val = tg.gradient(val)

# divergence of gradient
div_grad_val = tg.divergence(grad_val)

print(val.shape)
print(grad_val.shape)
print(div_grad_val.shape)
