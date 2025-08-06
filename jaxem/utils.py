import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def print_dict(dict):
    for key, value in dict.items():
        print(f"{key}:")
        print(f"    {value}")
    print("")

def latin_hypercube(key, n, d, minvals=0., maxvals=1.):
    """
    Generates n d-dimensional space-filling samples from a rectangular domain
    described by minvals and maxvals. By default, the domain is [0,1)^d.
    """
    key, key_in = jax.random.split(key)
    samples = jax.random.uniform(key_in, (n, d))/n + jnp.linspace(0, 1, n+1)[:n][:,jnp.newaxis]
    
    for i in range(d):
        key, key_in = jax.random.split(key)
        perm = jax.random.permutation(key_in, n, independent=True)
        samples = samples.at[:,i].set(samples[perm, i])
        
    minvals = jnp.array(minvals) * jnp.ones(d)
    maxvals = jnp.array(maxvals) * jnp.ones(d)
    samples = minvals + (maxvals - minvals) * samples
            
    return samples
    
    
def sample_LECs(key, Nsamples, LECs_best, scale_min=0., scale_max=2., static_indices=[]):

    LECs_lbd = scale_min * LECs_best
    LECs_ubd = scale_max * LECs_best
    
    for i in static_indices:
        LECs_lbd = LECs_lbd.at[i].set(LECs_best[i])
        LECs_ubd = LECs_ubd.at[i].set(LECs_best[i])

    key, key_in = jax.random.split(key)
    LECs_samples = latin_hypercube(key_in, Nsamples, LECs_best.shape[0], minvals=LECs_lbd, maxvals=LECs_ubd)

    return LECs_samples
    

    
def legendre(lmax, x):
    """
    Returns legendre polynomial P_l(x) and its derivative (dP_l/dx)(x)
    for all l up to lmax at all values in array x.
    """
    P, dP = jax.scipy.special.lpmn(lmax, lmax, x)
    return P[0], dP[0]
    

    
def grid(xmin, xmax, N, eps=1e-8):
    """
    Returns N Gauss points in [xmin, xmax]
    and their corresponding weights.
    """
    
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
    
    xo, xn, dP = jax.lax.while_loop(cond_func, body_func, (xo, xn, dP))
        
    x1 = 0.5 * (xmax - xmin)
    x2 = 0.5 * (xmax + xmin)
    x = x2 - x1 * xn
    w = 2. * x1 / ((1. - xn * xn) * dP * dP)

    return x, w

