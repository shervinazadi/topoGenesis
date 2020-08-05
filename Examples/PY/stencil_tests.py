import volpy as vp

print("runner")
# create a stencil based on well-known neighbourhood definitions
von_neumann_stencil = vp.create_stencil("von_neumann", 1, 1)

# create a moore neighbourhood stencil
moore_stencil = vp.create_stencil("moore", 1, 1)
moore_stencil.set_index([0,0,0], 0)
# print(moore_stencil.ntype)

custom_stencil = moore_stencil - von_neumann_stencil

# print(type(custom_stencil))
# print(custom_stencil)
# print(custom_stencil.ntype)
# print(custom_stencil.origin)
print(custom_stencil.bounds)
# print(custom_stencil.minbound)
# print(custom_stencil.maxbound)