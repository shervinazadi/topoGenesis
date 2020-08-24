import topogenesis as tg

print("runner")
# create a stencil based on well-known neighbourhood definitions
mc_stencil = tg.create_stencil("boolean_marching_cube", 1)
print(mc_stencil)
print(mc_stencil.ntype)
print(mc_stencil.origin)
print(mc_stencil.minbound)
print(mc_stencil.maxbound)

print(mc_stencil.expand())
print(mc_stencil.expand('F'))

# # create a moore neighbourhood stencil
# moore_stencil = tg.create_stencil("moore", 1, 1)
# moore_stencil.set_index([0,0,0], 0)
# # print(moore_stencil.ntype)

# custom_stencil = moore_stencil - von_neumann_stencil

# # print(type(custom_stencil))
# # print(custom_stencil)
# # print(custom_stencil.ntype)
# # print(custom_stencil.origin)
# print(custom_stencil.bounds)
# # print(custom_stencil.minbound)
# # print(custom_stencil.maxbound)