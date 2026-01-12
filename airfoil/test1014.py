from fealpy.backend import backend_manager as bm
#bm.set_backend('pytorch')
from fealpy.cfd import IncompressibleNSLFEM2DModel

from fealpy.decorator import cartesian, variantmethod
from fealpy.backend import TensorLike
from fealpy.mesh import TriangleMesh
import gmsh


class FlowPastFoil:
    '''
    @brief 圆柱绕流
    '''
    def __init__(self, eps=1e-10, rho=1, mu=0.001):
        self.eps = eps
        self.rho = rho
        self.mu = mu

    def mesh(self, box, h, hs=[0.01,0.005], method:str='fealpy', device='cpu'):
        from box_with_halo import generate_mesh
        self.box = box
        halos = [
            [1.0000,    0.00120],
            [0.9500,     0.01027],
            [0.9000,     0.01867],
            [0.8000,     0.03320],
            [0.7000,     0.04480],
            [0.6000,     0.05320],
            [0.5000,     0.05827],
            [0.4000,     0.06000],
            [0.3000,     0.05827],
            [0.2000,     0.05293],
            [0.1500,     0.04867],
            [0.1000,     0.04240],
            [0.0750,     0.03813],
            [0.0500,     0.03267],
            [0.0250,     0.02453],
            [0.0125,     0.01813],
            [0.0000,     0.00000],
            [0.0125,     -0.01813],
            [0.0250,     -0.02453],
            [0.0500,     -0.03267],
            [0.0750,     -0.03813],
            [0.1000,     -0.04240],
            [0.1500,     -0.04867],
            [0.2000,     -0.05293],
            [0.3000,     -0.05827],
            [0.4000,     -0.06000],
            [0.5000,     -0.05827],
            [0.6000,     -0.05320],
            [0.7000,     -0.04480],
            [0.8000,     -0.03320],
            [0.9000,     -0.01867],
            [0.9500,     -0.01027],
            [1.0000,     -0.00120]]
        singular_points = [[0, 0], [1, 0]]

        mesh = generate_mesh(box, halos, h, singular_points, hs, is_bspline=False)
        return mesh

    @cartesian
    def is_outflow_boundary(self,p):
        x = p[...,0]
        y = p[...,1]
        cond1 = bm.abs(x - self.box[1]) < self.eps
        cond2 = bm.abs(y-self.box[2])>self.eps
        cond3 = bm.abs(y-self.box[3])>self.eps
        return (cond1) & (cond2 & cond3) 
    
    @cartesian
    def is_inflow_boundary(self,p):
        return bm.abs(p[..., 0]-self.box[0]) < self.eps
     
    @cartesian
    def is_wall_boundary(self,p):
        return (bm.abs(p[..., 1] -self.box[2]) < self.eps) | \
               (bm.abs(p[..., 1] -self.box[3]) < self.eps)
    
    @cartesian
    def is_velocity_boundary(self,p):
        return ~self.is_outflow_boundary(p)
        # return None
    
    @cartesian
    def is_pressure_boundary(self,p=None):
        if p is None:
            return 1
        else:
            return self.is_outflow_boundary(p) 
            #return bm.zeros_like(p[...,0], dtype=bm.bool)
        # return 0

    @cartesian
    def u_inflow_dirichlet(self, p):
        x = p[...,0]
        y = p[...,1]
        value = bm.zeros_like(p)
        value[...,0] = 1.5*4*(y-self.box[2])*(self.box[3]-y)/(0.4**2)
        value[...,1] = 0
        return value
    
    @cartesian
    def pressure_dirichlet(self, p, t):
        x = p[...,0]
        y = p[...,1]
        value = bm.zeros_like(x)
        return value

    @cartesian
    def velocity_dirichlet(self, p, t):
        x = p[...,0]
        y = p[...,1]
        index = self.is_inflow_boundary(p)
        result = bm.zeros_like(p)
        result[index] = self.u_inflow_dirichlet(p[index])
        return result
    
    @cartesian
    def source(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = 0
        result[..., 1] = 0
        return result
    
    def velocity(self, p ,t):
        x = p[...,0]
        y = p[...,1]
        value = bm.zeros(p.shape)
        return value

    def pressure(self, p, t):
        x = p[..., 0]
        val = bm.zeros_like(x)
        return val



class  NACA0012Mesher:
    """
    A mesher for generating mesh of a NACA 0012 airfoil within a rectangular box.

    Parameters
    naca_points : array_like
        The coordinates of the NACA 0012 airfoil points,
        the point list must be continued,
        and the last point should be the trailing edge point.
    box : tuple
        The bounding box defined as (x_min, x_max, y_min, y_max).
    singular_points : array_like, optional
        Points where mesh refinement is needed, e.g., leading and trailing edges.
    """
    def __init__(self, naca_points:TensorLike, box=(-0.5, 1.5, -0.3, 0.3), singular_points:TensorLike=None):
        self.box = box
        self.naca_points = bm.array(naca_points, dtype=bm.float64)
        if singular_points is not None:
            self.singular_points = bm.array(singular_points, dtype=bm.float64)
        else:
            self.singular_points = None
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("naca0012")

        # 创建大矩形
        box_sphere = gmsh.model.occ.addRectangle(box[0], box[2], 0, box[1]-box[0], box[3]-box[2])
        # 创建 NACA 0012翼型
        point_tags = []
        for p in self.naca_points:
            point_tags.append(gmsh.model.occ.addPoint(p[0], p[1], 0))
        self.naca_points_tags = point_tags
        line_tags = []
        for i, p in enumerate(point_tags):
            # 首尾相连：最后一个点连接到第一个点
            p1 = point_tags[i]
            p2 = point_tags[(i + 1) % len(point_tags)]  # 模运算确保闭合
            line = gmsh.model.occ.addLine(p1, p2)
            line_tags.append(line)
        self.naca_line_tags = line_tags
        halo_curve_loop = gmsh.model.occ.addCurveLoop(line_tags)
        halo_surface = gmsh.model.occ.addPlaneSurface([halo_curve_loop])
        domain_tag, _ = gmsh.model.occ.cut([(2, box_sphere)], [(2, halo_surface)])
        gmsh.model.occ.synchronize()


    def geo_dimension(self) -> int:
        return 2

    @variantmethod('tri')
    def init_mesh(self, h=0.05, singular_h=None, is_quad = 0,
                  thickness=0.005, ratio=2.4, size=0.001) -> TriangleMesh:
        """
        Using Gmsh to generate a 2D triangular mesh for a NACA 0012 airfoil within a rectangular box.

        :param h: the global mesh size
        :param singular_h: the local mesh size at singular points
        :param is_quad: is the boundary layer mesh quadrilateral
        :param thickness: the thickness of the boundary layer
        :param ratio: the growth ratio of the boundary layer
        :param size: the initial size of the boundary layer
        :return: the triangle mesh of the NACA 0012 airfoil
        """
        # 设置边界层
        f = gmsh.model.mesh.field.add('BoundaryLayer')
        gmsh.model.mesh.field.setNumbers(f, 'CurvesList', self.naca_line_tags)
        gmsh.model.mesh.field.setNumber(f, 'Size', h / 50)
        gmsh.model.mesh.field.setNumber(f, 'Ratio', 2.4)
        gmsh.model.mesh.field.setNumber(f, 'Quads', is_quad)
        gmsh.model.mesh.field.setNumber(f, 'Thickness', h / 10)
        gmsh.option.setNumber('Mesh.BoundaryLayerFanElements', 7)
        gmsh.model.mesh.field.setNumbers(f, 'FanPointsList', [self.naca_points_tags[-1]])
        gmsh.model.mesh.field.setAsBoundaryLayer(f)
        if self.singular_points is not None:
            if singular_h is None:
                singular_h = [h/10]*len(self.singular_points)
            elif len(singular_h) != len(self.singular_points):
                raise ValueError("Length of singular_h must match number of singular_points.")
            # 创建奇异点
            singular_point_tags = []
            for i, p in enumerate(self.singular_points):
                singular_point_tags.append(gmsh.model.occ.addPoint(p[0], p[1], 0, singular_h[i]))
            gmsh.model.occ.synchronize()
            # 设置背景网格
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
            # 为奇异点设置局部网格加密
            fields = []  # 收集所有 Threshold Field
            for i, sp in enumerate(singular_point_tags):
                f_dist = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(f_dist, "PointsList", [sp])

                f_thresh = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
                gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", singular_h[i])
                gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", h)
                gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", (self.box[3]-self.box[2])/100)
                gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", (self.box[3]-self.box[2])/2)  # 缩小加密范围

                fields.append(f_thresh)

            # 创建 Min Field 合并所有 Threshold Field
            min_field = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", fields)
            # 设置背景网格
            gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
        else:
            # 设置背景网格
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)

        gmsh.model.mesh.generate(2)
        # gmsh.fltk.run()
        node_tags, node, _ = gmsh.model.mesh.getNodes()
        node = bm.array(node, dtype=bm.float64).reshape(-1, 3)[:, :2]
        element_types, element_tags, cell = gmsh.model.mesh.getElements(2)
        cell = bm.array(cell[0], dtype=bm.int64).reshape(-1, 3) - 1

        gmsh.finalize()
        return TriangleMesh(node, cell)


# if __name__ == '__main__':
#     box = [-0.5, 2.7, -0.5, 0.5]
#     h = 0.03
#     halos = bm.array([
#         [1.0000, 0.00120],
#         [0.9500, 0.01027],
#         [0.9000, 0.01867],
#         [0.8000, 0.03320],
#         [0.7000, 0.04480],
#         [0.6000, 0.05320],
#         [0.5000, 0.05827],
#         [0.4000, 0.06000],
#         [0.3000, 0.05827],
#         [0.2000, 0.05293],
#         [0.1500, 0.04867],
#         [0.1000, 0.04240],
#         [0.0750, 0.03813],
#         [0.0500, 0.03267],
#         [0.0250, 0.02453],
#         [0.0125, 0.01813],
#         [0.0000, 0.00000],
#         [0.0125, -0.01813],
#         [0.0250, -0.02453],
#         [0.0500, -0.03267],
#         [0.0750, -0.03813],
#         [0.1000, -0.04240],
#         [0.1500, -0.04867],
#         [0.2000, -0.05293],
#         [0.3000, -0.05827],
#         [0.4000, -0.06000],
#         [0.5000, -0.05827],
#         [0.6000, -0.05320],
#         [0.7000, -0.04480],
#         [0.8000, -0.03320],
#         [0.9000, -0.01867],
#         [0.9500, -0.01027],
#         [1.0000, -0.00120],
#         [1.00662, 0.0]], dtype=bm.float64)
#     singular_points = bm.array([[0, 0], [1.00662, 0.0]], dtype=bm.float64)
#     hs = [h/2, h]  # 奇异点处网格尺寸

#     mesher = NACA0012Mesher(halos, box, singular_points)
#     mesh = mesher.init_mesh(h, hs, is_quad=0, thickness=h/10, ratio=2.4, size=h/50)
#     mesh.to_vtk(fname='naca0012_tri.vtu')


pde = FlowPastFoil(rho=1, mu=0.001)

box = [-0.5, 2.7, -0.4, 0.4]
hs = [0.02, 0.02]  # 奇异点处网格尺寸

mesh = pde.mesh(box, h=0.04, hs=hs)

from matplotlib import pyplot as plt
fig, axes = plt.subplots()
mesh.add_plot(axes)
plt.show()



print(mesh.number_of_cells())



options ={
    'backend': 'numpy',
    'pde': 2,
    'rho': 1.0,
    'mu': 0.001,
    'T0': 0.0,
    'T1': 1.0,
    'nt': 10000,
    'init_mesh': 'tri',
    'center': (0.2, 0.2),
    'radius': 0.05,
    'n_circle': 300,
    'lc': 0.05,
    'method': 'IPCS',
    'solve': 'direct',
    'apply_bc': 'cylinder',
    'postprocess': 'res',
    'run': 'main_cylinder',
    'maxit': 5,
    'maxstep': 10,
    'tol': 1e-10
}


model = IncompressibleNSLFEM2DModel(pde=pde, mesh = mesh, options = options)
model.equation.set_constitutive(1)
model.equation.set_coefficient('viscosity', pde.mu)

mesh = model.mesh         
pde = model.pde
fem = model.fem
fem.dt = model.timeline.dt
maxstep = model.maxstep 
tol = model.tol 

u0 = fem.uspace.interpolate(cartesian(lambda p: pde.velocity(p, model.timeline.T0)))
p0 = fem.pspace.interpolate(cartesian(lambda p: pde.pressure(p, model.timeline.T0)))
cd = bm.zeros(model.timeline.NL-1)
cl = bm.zeros(model.timeline.NL-1)
delta_p = bm.zeros(model.timeline.NL-1)

mesh.nodedata['ph'] = p0
mesh.nodedata['uh'] = u0.reshape(model.mesh.GD,-1).T
mesh.to_vtk(f'air_foil_{str(0).zfill(10)}.vtu')
for i in range(model.timeline.NL-1):
    t  = model.timeline.current_time()
    model.logger.info(f"time={t}")
    
    u1,p1 = model.run['one_step'](u0, p0, maxstep, tol)
    u0[:] = u1
    p0[:] = p1

    if i < 20000 :
        mesh.nodedata['ph'] = p1
        mesh.nodedata['uh'] = u1.reshape(model.mesh.GD,-1).T
        mesh.to_vtk(f'ns2d_{str(i+1).zfill(10)}.vtu')

    model.timeline.advance()



