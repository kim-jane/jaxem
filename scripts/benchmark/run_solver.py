import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import time
from jaxem import *


# instantiate objects
config = Config('scripts/benchmark/config.ini')
mesh = TRNS(n_mesh=config.n_mesh)
channels = Channels(isospin_channel=config.isospin_channel, Jmax=config.Jmax)
potential = Chiral()
solver = Solver(mesh, channels, potential, config.Elabs)


# solve the Lippmann-Schwinger equation for the onshell t-matrix
# for all channels and energies at best-fit LECs for Chiral potential
t_onshell = solver.onshell_t(potential.LECs)


# compute total cross sections and phase shifts
sigma_tot, _, single_output, coupled_output = solver.scattering_params((t_onshell, None))
delta_ck, eta_ck, sigma_ck = single_output
delta_minus, delta_plus, epsilon, eta_minus, eta_plus, sigma_cck = coupled_output

print(sigma_tot.shape)
print(single_output.shape)
print(coupled_output.shape)

# write results to file
jnp.savez(
    'scripts/benchmark/solver_output.npz',
    sigma_tot=sigma_tot,
    single_output=single_output,
    coupled_output=coupled_output
)

