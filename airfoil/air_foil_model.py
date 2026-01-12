#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: navier_stokes_equation_2d.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Mon 14 Oct 2024 04:53:51 PM CST
	@bref 
	@ref 
'''  
from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh

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
    
    @cartesian
    def is_pressure_boundary(self,p=None):
        if p is None:
            return 1
        else:
            return self.is_outflow_boundary(p) 
            #return bm.zeros_like(p[...,0], dtype=bm.bool)

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
