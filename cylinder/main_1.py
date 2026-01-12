from fealpy.backend import backend_manager as bm
from fealpy.cfd import IncompressibleNSLFEM2DModel
from fealpy.pde.navier_stokes_equation_2d import FlowPastCylinder as pde


pde = pde(rho=1, mu=0.001)
mesh = pde.mesh(h=0.05)

model = IncompressibleNSLFEM2DModel(pde, mesh)

timeline = model.timeline
timeline.set_timeline(0,1,2000)

equation = model.equation
equation.set_constitutive(1)
equation.set_coefficient('viscosity', pde.mu)

fem = model.fem

#model.method['Newton']()
model.method['IPCS']()
model.run['main'](maxstep=5, tol=1e-10)



