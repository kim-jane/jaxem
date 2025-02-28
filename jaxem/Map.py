from .Tools import grid
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)


class Map:

    def __init__(self, config):
    
        self.Nq = config.Nq  # total number of mesh points
        
        if config.map == 'tan':
            self.grid = self.grid_tan
            self.c = config.c  # fm^-1
            self.inf = True
            
        elif config.map == 'trns':
        
            self.Nq1 = config.Nq1
            self.Nq2 = self.Nq - self.Nq1
            self.p1 = config.p1 # fm^-1
            self.p2 = config.p2 # fm^-1
            self.p3 = config.p3 # fm^-1
            self.grid = self.grid_trns
            self.inf = False
            
        else:
            raise ValueError(f"Map {config.map} unknown.")
    
        

    def grid_tan(self):
    
        def map(k):
            return self.c * jnp.tan(0.5 * jnp.pi * k)
    
        def dmap(k):
            sec = 1. / jnp.cos(0.5 * jnp.pi * k)
            return 0.5 * jnp.pi * self.c * sec * sec
    
        q, wq = grid(0., 1., self.Nq)
        wq = dmap(q) * wq
        q = map(q)
        return q, wq


    def grid_trns(self):
    
        def map1(k):
            return (1. + k) / (1./self.p1 - (1./self.p1 - 2./self.p2) * k)
            
        def dmap1(k):
            num = -2. * self.p1 * self.p2 * (self.p1 - self.p2)
            denom = self.p2 * (k - 1.) - 2. * self.p1 * k
            return num / denom**2
            
        def map2(k):
            return 0.5 * (self.p2 + self.p3) + 0.5 * (self.p3 - self.p2) * k
            
        def dmap2(k):
            return 0.5 * (self.p3 - self.p2)
            
        q1, wq1 = grid(-1., 1., self.Nq1)
        wq1 = dmap1(q1) * wq1
        q1 = map1(q1)
        
        q2, wq2 = grid(-1., 1., self.Nq2)
        wq2 = dmap2(q2) * wq2
        q2 = map2(q2)
        
        q = jnp.concatenate((q1, q2))
        wq = jnp.concatenate((wq1, wq2))
        
        return q, wq
