from .utils import grid
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)


class Mesh:

    def __init__(
        self,
        n_mesh: int = 40,
    ):
        self.n_mesh = n_mesh
        
        
        
class Tangent(Mesh):

    def __init__(
        self,
        n_mesh: int = 40,
        c: float = 5.0,      # fm^-1
    ):
    
        super().__init__(n_mesh)
        self.c = c
        self.inf = True

        def map(x):
            return self.c * jnp.tan(0.5 * jnp.pi * x)
    
        def dmap(x):
            sec = 1. / jnp.cos(0.5 * jnp.pi * x)
            return 0.5 * jnp.pi * self.c * sec * sec
    
        q, wq = grid(0., 1., self.Nq)
        self.wq = dmap(q) * wq # fm^-1
        self.q = map(q) # fm^-1
        
        
class TRNS(Mesh):

    def __init__(
        self,
        n_mesh: int = 40,
        frac: float = 0.8,   # fraction of grid points in [0, p2]
        p1: float = 2.5,     # fm^-1
        p2: float = 10.0,     # fm^-1
        p3: float = 30.0     # fm^-1
    ):
    
        super().__init__(n_mesh)
        self.n_mesh1 = int(frac * n_mesh)
        self.n_mesh2 = self.n_mesh - self.n_mesh1
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.inf = False
        
        def map1(x):
            return (1. + x) / (1./self.p1 - (1./self.p1 - 2./self.p2) * x) # fm^-1
            
        def dmap1(x):
            num = -2. * self.p1 * self.p2 * (self.p1 - self.p2)
            denom = self.p2 * (x - 1.) - 2. * self.p1 * x
            return num / denom**2 # fm^-1
            
        def map2(x):
            return 0.5 * (self.p2 + self.p3) + 0.5 * (self.p3 - self.p2) * x # fm^-1
            
        def dmap2(x):
            return 0.5 * (self.p3 - self.p2) # fm^-1
            
        q1, wq1 = grid(-1., 1., self.n_mesh1)
        wq1 = dmap1(q1) * wq1
        q1 = map1(q1)
        
        q2, wq2 = grid(-1., 1., self.n_mesh2)
        wq2 = dmap2(q2) * wq2
        q2 = map2(q2)
        
        self.q = jnp.concatenate((q1, q2))
        self.wq = jnp.concatenate((wq1, wq2))
        
