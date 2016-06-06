from domains_and_boundaries import *
from numpy import zeros, where, linspace, ones, array, loadtxt
import scipy.interpolate as Spline
import sys
set_log_active(False)
parameters['allow_extrapolation']=True


RL = '1'
EkPa = '62500'

ref_level = int(RL)
E = Constant(float(EkPa))

print 'ref: ',RL
print 'E: ',EkPa

ufile = File("RESULTS/reversed%s_E%s_3mm/velocity.pvd"%(RL,EkPa)) # xdmf
pfile = File("RESULTS/reversed%s_E%s_3mm/pressure.pvd"%(RL,EkPa))
dfile = File("RESULTS/reversed%s_E%s_3mm/dU.pvd"%(RL,EkPa))
tfile = File("RESULTS/reversed%s_E%s_3mm/U.pvd"%(RL,EkPa))


# PHYSICAL PARAMETERS
# FLUID
rho_f = Constant(1./1000)		# [g/mm]
nu_f = Constant(0.658)			# [mm**2/s]
mu_f = Constant(nu_f*rho_f)		# [g/(mm*s)]

# SOLID
Pr = 0.479
rho_s = Constant(1.75*rho_f)
lamda = Constant(E*Pr/((1.0+Pr)*(1.0-2*Pr)))
mu_s = Constant(E/(2*(1.0+Pr)))


Nx = 18*ref_level
Ny = 30*ref_level
P1 = Point(-9,0)
P2 = Point(9,60)
mesh = RectangleMesh(P1,P2,Nx,Ny)
LEN = len(mesh.coordinates())

SD = MeshFunction('size_t', mesh, mesh.topology().dim())
SD.set_all(0)
Solid().mark(SD,1)
#CSC().mark(SD,1)  #Remove comment for solid cord


# DEFINING BOUNDARIES
boundaries = FacetFunction("size_t",mesh)
boundaries.set_all(0)
Fluid_in_l().mark(boundaries,1)
Fluid_in_r().mark(boundaries,2)
Solid_in().mark(boundaries,3)
Fluid_out().mark(boundaries,4)
Solid_out().mark(boundaries,5)
Interface().mark(boundaries,6)
Fluid_walls().mark(boundaries,7)
CSC_bnd().mark(boundaries,6)


dt = 0.002
T = 5

# TEST AND TRIALFUNCTIONS
V = VectorFunctionSpace(mesh,'CG',2)
P = FunctionSpace(mesh,'CG',1)
W = VectorFunctionSpace(mesh,'CG', 1)
VPW = MixedFunctionSpace([V,P,W])
print 'dofs: ' ,VPW.dim()
v,p,w = TrialFunctions(VPW)
phi,eta,psi = TestFunctions(VPW)



# INITIAL AND BOUNDARY CONDITIONS

# FLUID
noslip = Constant((0.0,0.0))

t_pres,pres = np.loadtxt('Eide_normalized.txt')
pres -= 0.12   # pres -= 0.2 gives too much cranial
pres *= -1
pres -= 0.0167
pres = np.append(pres,pres[1])
t_pres = np.append(t_pres,t_pres[-1]+t_pres[1])
applied_p = Spline.PchipInterpolator(t_pres,pres)

class applied_pres(Expression):
	def __init__(self):
		self.t = 0
		self.amp = 1
	def eval(self,value,x):
		period = t_pres[-2]
		cycle_t = self.t
		while cycle_t>=period:
			if cycle_t > period:
				cycle_t -= period
		
		value[0] = self.amp*applied_p(cycle_t)


pressure = Expression(('amp*sin(2*pi*t)'),t=0,amp=1)

bcv3 = DirichletBC(VPW.sub(0),noslip,boundaries,3) # Solid in
bcv4 = DirichletBC(VPW.sub(0),noslip,boundaries,4) # Fluid out
bcv5 = DirichletBC(VPW.sub(0),noslip,boundaries,5) # Solid out
bcv6 = DirichletBC(VPW.sub(0),noslip,boundaries,6) # Interface
bcv7 = DirichletBC(VPW.sub(0),noslip,boundaries,7) # Fluid walls


bcv = [bcv3, bcv5, bcv7]


# SOLID

# MESH DISPLACEMENT

bcw1 = DirichletBC(VPW.sub(2),noslip,boundaries,1)  # Fluid in_l
bcw2 = DirichletBC(VPW.sub(2),noslip,boundaries,2)  # Fluid in_r
bcw3 = DirichletBC(VPW.sub(2),noslip,boundaries,3)  # Solid in
bcw4 = DirichletBC(VPW.sub(2),noslip,boundaries,4)  # Fluid out
bcw5 = DirichletBC(VPW.sub(2),noslip,boundaries,5)  # Solid out
bcw6 = DirichletBC(VPW.sub(2),noslip,boundaries,6)  # Interface
bcw7 = DirichletBC(VPW.sub(2),noslip,boundaries,7) # Fluid walls
bcw = [bcw1,bcw2,bcw3,bcw4,bcw5,bcw7]

# CREATE FUNCTIONS
v0 = Function(V)
v1 = Function(V)
U1 = Function(W)

VPW_ = Function(VPW)

k = Constant(dt)
n = FacetNormal(mesh)

dS = Measure('dS')[boundaries]
dx = Measure('dx')[SD]
ds = Measure('ds')[boundaries]

dx_f = dx(0,subdomain_data=SD)
dx_s = dx(1,subdomain_data=SD)


def sigma_dev(U):
	return 2*mu_s*sym(grad(U)) + lamda*tr(sym(grad(U)))*Identity(2)


delta = 1e-8

# VARIATIONAL FORMULATION

# SOLID
U = U1 + k*v      # Implicit Displacement scheme (v is used to satisfy BB)

D = inner(sigma_dev(U),grad(phi))*dx_s

aMS = rho_s/k*inner(v,phi)*dx_s \
	+ rho_s*inner(grad(v0)*v,phi)*dx_s \
	+ lhs(D)

LMS = rho_s/k*inner(v1,phi)*dx_s \
	+ rhs(D)

aDS = 1/delta*inner(v,psi)*dx_s - 1/delta*inner(w,psi)*dx_s



aS = aMS + aDS
LS = LMS

# FLUID
penalty = 0.01*mesh.hmin()


aMF = rho_f/k*inner(v,phi)*dx_f \
	+ rho_f*inner(grad(v0)*(v-w),phi)*dx_f \
	- inner(p,div(phi))*dx_f \
	+ 2*mu_f*inner(sym(grad(v)),grad(phi))*dx_f \
	- mu_f*inner(grad(v).T*n,phi)*ds(1) \
	- mu_f*inner(grad(v).T*n,phi)*ds(2) \
	- mu_f*inner(grad(v).T*n,phi)*ds(4) \
	+ penalty**-2*(inner(v,phi)-inner(dot(v,n),dot(phi,n)))*ds(1) \
	+ penalty**-2*(inner(v,phi)-inner(dot(v,n),dot(phi,n)))*ds(2) \
	+ penalty**-2*(inner(v,phi)-inner(dot(v,n),dot(phi,n)))*ds(4)

LMF = rho_f/k*inner(v1,phi)*dx_f - \
	inner(pressure*n,phi)*ds(1) - \
	inner(pressure*n,phi)*ds(2)

aDF = k*inner(grad(w),grad(psi))*dx_f \
	- k*inner(grad(w('-'))*n('-'),psi('-'))*dS(6)
LDF = -inner(grad(U1),grad(psi))*dx_f \
	+ inner(grad(U1('-'))*n('-'),psi('-'))*dS(6)

aCF = -inner(div(v),eta)*dx_f

aF = aMF + aDF + aCF
LF = LMF + LDF

# ADD LINEAR AND BILINEAR FORMS
a = aS+aF
L = LS+LF

t = dt

count = 0

# TIME LOOP
while t < T + DOLFIN_EPS:

	if t < 1:
		pressure.amp = 10*t
	pressure.t = t
	b = assemble(L)
	eps = 10
	k_iter = 0
	max_iter = 5
	while eps > 1E-6 and k_iter < max_iter:
	    A = assemble(a)
	    A.ident_zeros()
	    [bc.apply(A,b) for bc in bcv]
	    [bc.apply(A,b) for bc in bcw]
	    solve(A,VPW_.vector(),b,'lu')
	    v_,p_,w_ = VPW_.split(True)
	    eps = errornorm(v_,v0,degree_rise=3)
	    k_iter += 1
	    print 'k: ',k_iter, 'error: %.3e' %eps
	    v0.assign(v_)
	if count%5==0:		# save every fifth state
		ufile << v_
		pfile << p_
		dfile << w_
		tfile << U1

	
	w_.vector()[:] *= float(k)
	U1.vector()[:] += w_.vector()[:]
	ALE.move(mesh,w_)
	mesh.bounding_box_tree().build(mesh)

	# Move to next time step
	v1.assign(v_)
	print 't=%.4f'%t
	t += dt
	count += 1
