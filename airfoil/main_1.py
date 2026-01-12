from fealpy.backend import backend_manager as bm
#bm.set_backend('pytorch')
from fealpy.cfd import IncompressibleNSLFEM2DModel
from air_foil_model import FlowPastFoil as pde

pde = pde(rho=1, mu=0.001)
box = [-0.5, 2.7, -0.5, 0.5]
hs = [0.005, 0.001]  # 奇异点处网格尺寸

mesh = pde.mesh(box, h=0.04, hs=hs)
mesh.uniform_refine(n=1)

from matplotlib import pyplot as plt
fig, axes = plt.subplots()
mesh.add_plot(axes)
plt.show()

model = IncompressibleNSLFEM2DModel(pde, mesh)

timeline = model.timeline
timeline.set_timeline(0,1,10000)

equation = model.equation
equation.set_constitutive(1)
equation.set_coefficient('viscosity', pde.mu)

fem = model.fem

model.method['Ossen']()
#model.method['IPCS']()
model.run['main'](maxstep=5, tol=1e-10)



