"""
This script solves the Lippmann-Schwinger 
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
from jaxem import *
from collections import defaultdict
import corner


print("=" * 100)
print("SOLVER")
print("=" * 100)

Jmax = 6
Elabs = [10., 50., 100., 200.]


for n_mesh in [20, 40, 60, 80, 100, 120]:

    print("=" * 100)
    print("MESH POINTS = ", n_mesh)
    print("=" * 100)

    mesh = TRNS(n_mesh=n_mesh)
    channels = Channels(isospin_channel='np', Jmax=Jmax)
    potential = Chiral()
    solver = Solver(mesh, channels, potential, Elabs)

    # warm up jit-compilation
    for i in range(10):

        t_init = time.time()
        t_and_err_onshell = solver.onshell_t_and_err(potential.LECs)
        sigma_tot, err_sigma_tot, single_output, coupled_output = solver.scattering_params(t_and_err_onshell)
        delta_ck, eta_ck, sigma_ck = single_output
        delta_minus, delta_plus, epsilon, eta_minus, eta_plus, sigma_cck = coupled_output
        #print(f"{i}, {time.time() - t_init}")

    # print times
    n_iters = 100

    times = jnp.zeros(n_iters)
    for i in range(n_iters):

        t_init = time.time()
        t_and_err_onshell = solver.onshell_t_and_err(potential.LECs)
        sigma_tot, err_sigma_tot, single_output, coupled_output = solver.scattering_params(t_and_err_onshell)
        delta_ck, eta_ck, sigma_ck = single_output
        delta_minus, delta_plus, epsilon, eta_minus, eta_plus, sigma_cck = coupled_output
        t_final = time.time()
        
        #print(f"{i}, {t_final - t_init}")
        times = times.at[i].set(t_final - t_init)
    
    print(f"\n\n!*! {n_mesh}, {jnp.mean(times)}, {jnp.std(times, ddof=1)/jnp.sqrt(n_iters)}")
