from fealpy.backend import backend_manager as bm
from fealpy.pde.navier_stokes_equation_2d import FlowPastCylinder as pde
from fealpy.cfd.simulation.fem.incompressible_ns import IPCS,Newton,Ossen
from fealpy.cfd.equation import IncompressibleNS
from fealpy.cfd.simulation.time import UniformTimeLine
from fealpy.fem.dirichlet_bc import DirichletBC
from fealpy.decorator import cartesian
from fealpy.solver import spsolve

pde = pde(rho=1, mu=0.001)
mesh = pde.mesh(h=0.05)

import matplotlib.pyplot as plt
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
#plt.show()

equation = IncompressibleNS(pde, init_variables=True)
equation.set_constitutive(1)
#IPCS = IPCS(equation, mesh)
newton = Newton(equation, mesh)
#newton = Ossen(equation, mesh)
timeline = UniformTimeLine(0, 1, 1000)
newton.dt = timeline.dt

Bform = newton.BForm()
Lform = newton.LForm()

u = newton.uspace.function()
uk0 = newton.uspace.function()
uk1 = newton.uspace.function()

p = newton.pspace.function()
pk0 = newton.pspace.function()
pk1 = newton.pspace.function()

maxit = 5
tol = 1e-10
ugdof = newton.uspace.number_of_global_dofs()

for i in range(timeline.NL-1):
    t = timeline.current_time()
    print(f'================== i = {i}, time = {t:.4f} ====================')
    equation.set_coefficient('body_force', cartesian(lambda p: pde.source(p, timeline.next_time())))  
    uk0[:] = u[:]
    pk0[:] = p[:]
    for j in range(maxit):
        newton.update(uk0, u)
        A = Bform.assembly()
        b = Lform.assembly()
        A,b = newton.apply_bc(A, b, pde, t=timeline.next_time())
        #A,b = BC.apply(A,b)
        print(bm.sum(bm.abs(A.toarray())))

        x = spsolve(A, b, 'mumps')
        uk1[:] = x[:ugdof]
        pk1[:] = x[ugdof:]
        res_u = mesh.error(uk0, uk1)
        res_p = mesh.error(pk0, pk1)
        print("res_u, res_p",res_u, res_p)
        uk0[:] = uk1
        pk0[:] = pk1
        if res_u+res_p < tol:
            break

    u[:] = uk1
    p[:] = pk1
    mesh.nodedata['ph'] = p
    mesh.nodedata['uh'] = u.reshape(2,-1).T
    mesh.to_vtk(f'ns2d_{i+1}.vtu')
    timeline.advance()
