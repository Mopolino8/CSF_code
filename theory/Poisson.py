from dolfin import *

mesh = UnitSquareMesh(10,10)

V = VectorFunctionSpace(mesh,'CG',2)
f = Constant((1.0,-1.0))

u = TrialFunction(V)
v = TestFunction(V)

LHS = inner(grad(u),grad(v))*dx
RHS = inner(f,v)*dx

u0 = Constant((0,0))
def boundary(x,on_bnd):
	return on_bnd

bcs = DirichletBC(V,u0,boundary)

u_ = Function(V)

solve(LHS==RHS,u_,bcs)

plot(u_)
interactive()
