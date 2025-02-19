import jax.numpy as jnp
import jax.scipy.special as jsp
from jax import jit, random, vmap
from jax.lax import while_loop, fori_loop
from jax import config
config.update("jax_enable_x64", True)
import time
from matplotlib import pyplot as plt

def latin_hypercube(key, n, d, minvals=0., maxvals=1.):
    """
    Generates n d-dimensional space-filling samples from a rectangular domain
    described by minvals and maxvals. By default, the domain is [0,1)^d.
    """
    key, key_in = random.split(key)
    samples = random.uniform(key_in, (n, d))/n + jnp.linspace(0, 1, n+1)[:n][:,jnp.newaxis]
    
    for i in range(d):
        key, key_in = random.split(key)
        perm = random.permutation(key_in, n, independent=True)
        samples = samples.at[:,i].set(samples[perm, i])
        
    minvals = jnp.array(minvals) * jnp.ones(d)
    maxvals = jnp.array(maxvals) * jnp.ones(d)
    samples = minvals + (maxvals - minvals) * samples
            
    return samples


    
def legendre(lmax, x):
    """
    Returns legendre polynomial P_l(x) and its derivative (dP_l/dx)(x)
    for all l up to lmax at all values in array x.
    """
    P, dP = jsp.lpmn(lmax, lmax, x)
    return P[0], dP[0]
    

    
def grid(xmin, xmax, N):
    """
    Returns N Gauss points in [xmin, xmax]
    and their corresponding weights.
    """
        
    eps = 1.0e-6
    
    def get_new_estimate(xo):
        P, dP = legendre(N, xo)
        xn = xo - P[-1] / dP[-1]
        return xn, dP[-1]

    def body_func(vals):
        xo, xn, dP = vals
        xo = xn
        xn, dP = get_new_estimate(xo)
        return xo, xn, dP
        
    def cond_func(vals):
        xo, xn, dP = vals
        return jnp.max(jnp.abs(xn - xo)) > eps
    
    i = jnp.arange(1, N+1)
    xo = jnp.cos(jnp.pi * (i - 0.25) / (N + 0.5))
    xn, dP = get_new_estimate(xo)
    
    xo, xn, dP = while_loop(cond_func, body_func, (xo, xn, dP))
        
    x1 = 0.5 * (xmax - xmin)
    x2 = 0.5 * (xmax + xmin)
    x = x2 - x1 * xn
    w = 2. * x1 / ((1. - xn * xn) * dP * dP)

    return x, w
