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
    
    
def sample_LECs(key, Nsamples, LECs_best, scale_min=0., scale_max=2., static_indices=[]):

    LECs_lbd = scale_min * LECs_best
    LECs_ubd = scale_max * LECs_best
    
    for i in static_indices:
        LECs_lbd = LECs_lbd.at[i].set(LECs_best[i])
        LECs_ubd = LECs_ubd.at[i].set(LECs_best[i])

    key, key_in = random.split(key)
    LECs_samples = latin_hypercube(key_in, Nsamples, LECs_best.shape[0], minvals=LECs_lbd, maxvals=LECs_ubd)

    return LECs_samples
    


def gram_schmidt_insert(Q, new_col):
    
    v = jnp.dot(Q, jnp.dot(jnp.conjugate(Q.T), new_col))
    
    u = new_col - v
    
    norm_u = jnp.linalg.norm(u)
    print("* norm_u = ", norm_u)
    q_new = u / norm_u
    
    # update the Q matrix by appending the new q_{n+1} column
    Q_new = jnp.hstack([Q, q_new.reshape(-1, 1)])
    
    return Q_new
    
def modified_gram_schmidt_insert(Q, new_col):

    u = new_col.copy()
    for i in range(Q.shape[1]):
        print(f"{i} {jnp.linalg.norm(Q[:, i]):.15e}")
        proj = jnp.dot(Q[:, i].conj(), u)
        u = u - proj * Q[:, i]
        u = u / jnp.linalg.norm(u) # just seeing if this will be more stable
    
    Q_new = jnp.hstack([Q, u.reshape(-1, 1)])
    
    for j in range(Q_new.shape[1]):
        for i in range(j+1):
            print(i, j, Q_new[:,i].conj().T @ Q_new[:,j])
    
    return Q_new
    
def householder_insert(Q, v): # there's a bug

    for i in range(Q.shape[1]):
        for j in range(i, Q.shape[1]):
            print(i, j, Q[:,i].conj().T @ Q[:,j])

    m, k = Q.shape
    
    # Project v onto the existing subspace
    v_proj = Q @ (Q.conj().T @ v)
    v_perp = v - v_proj
    
    # If v_perp is nearly zero, return original Q (indicating dependence)
    norm_v_perp = jnp.linalg.norm(v_perp)
    if norm_v_perp < 1e-12:
        raise ValueError("Inserted vector is nearly dependent on the existing basis.")
    
    # Householder vector: u = v_perp + sign(v_perp[0]) * ||v_perp|| * e_1
    sign_v0 = jnp.exp(1j * jnp.angle(v_perp[0])) if v_perp[0] != 0 else 1.0
    u = v_perp + sign_v0 * norm_v_perp * jnp.eye(m, 1)[:, 0]  # e_1 is the first basis vector
    u = u / jnp.linalg.norm(u)  # Normalize
    
    # Construct Householder reflection matrix: H = I - 2 uu^H
    H = jnp.eye(m) - 2 * jnp.outer(u, u.conj())
    
    # Apply reflection to v_perp to make it fully orthogonal
    q_new = H @ v_perp / jnp.linalg.norm(v_perp)
    #q_new = q_new / jnp.linalg.norm(q_new)
    # Append to Q
    Q_new = jnp.hstack([Q, q_new[:, None]])
    
    for i in range(Q_new.shape[1]):
        for j in range(i, Q_new.shape[1]):
            print(i, j, Q_new[:,i].conj().T @ Q_new[:,j])
            
    for i in range(Q_new.shape[1]):
        plt.plot(Q_new[:,i].real)
        plt.plot(Q_new[:,i].imag, linestyle='dashed')
    plt.show()
    
    return Q_new





    
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
        
    eps = 1.0e-8
    
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
