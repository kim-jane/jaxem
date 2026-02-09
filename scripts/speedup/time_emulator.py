import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import time
from jaxem import *


print("=" * 100)
print("EMULATOR")
print("=" * 100)

Jmax = 6
Elabs = [10., 50., 100., 200.]

for tol in [1e-6, 1e-7]:

    print("=" * 100)
    print("TOLERANCE = ", tol)
    print("=" * 100)

    for rom in ['lspg']:
    
        print("=" * 100)
        print("ROM = ", rom)
        print("=" * 100)
        
        for n_mesh in [20, 40, 60, 80, 100, 120]:

            print("=" * 100)
            print("MESH POINTS = ", n_mesh)
            print("=" * 100)

            mesh = TRNS(n_mesh)
            channels = Channels(isospin_channel='np', Jmax=Jmax)
            potential = Chiral()
            solver = Solver(mesh, channels, potential, Elabs)
            emulator = Emulator(solver)
            
            key = jax.random.PRNGKey(317)
            key, subkey = jax.random.split(key)
            LECs_candidates = sample_LECs(
                subkey,
                500,
                potential.LECs,
                scale_min=0.0,
                scale_max=2.0,
                static_indices=[0,10,11],
            )
            print(LECs_candidates.shape)

            # train emulator
            emulator.fit(LECs_candidates, rom='lspg', mode='greedy', n_init=2, n_max=100, tol=tol)
            
            # warm up jit-compilation
            for i in range(10):

                t_init = time.time()
                t_and_err_onshell = emulator.onshell_t_and_err(potential.LECs)
                sigma_tot, err_sigma_tot, single_output, coupled_output = solver.scattering_params(t_and_err_onshell)
                delta_ck, eta_ck, sigma_ck = single_output
                delta_minus, delta_plus, epsilon, eta_minus, eta_plus, sigma_cck = coupled_output
                #print(f"{i}, {time.time() - t_init}")

            # print times
            n_iters = 100

            times = jnp.zeros(n_iters)
            for i in range(n_iters):

                t_init = time.time()
                t_and_err_onshell = emulator.onshell_t_and_err(potential.LECs)
                sigma_tot, err_sigma_tot, single_output, coupled_output = solver.scattering_params(t_and_err_onshell)
                delta_ck, eta_ck, sigma_ck = single_output
                delta_minus, delta_plus, epsilon, eta_minus, eta_plus, sigma_cck = coupled_output
                t_final = time.time()
                
                #print(f"{i}, {t_final - t_init}")
                times = times.at[i].set(t_final - t_init)
            
            print(f"\n\n!*! {n_mesh}, {jnp.mean(times)}, {jnp.std(times, ddof=1)/jnp.sqrt(n_iters)}")

