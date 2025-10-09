import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
from jaxem import *
from collections import defaultdict
import corner
print(jax.__version__)


# data
filename = "benchmark/np_SGT/PWA93_0.1_300.txt"
Elabs, sigmas = np.loadtxt(filename, unpack=True)
indices = [0,13,18,28,38,48,58,68,78,88,98,108]
#indices = [0,5,10,20,30,40,50,60,70,80,90,100]
Elabs = Elabs[indices]
sigmas = sigmas[indices]



#############

n_mesh = 40
Jmax = 4
static_indices = [0, 10, 11]

mesh = TRNS(n_mesh=n_mesh)
channels = Channels(isospin_channel='np', Jmax=Jmax)
potential = Chiral()
solver = Solver(mesh, channels, potential, Elabs)
emulator = Emulator(solver)


# train emulator
key = jax.random.PRNGKey(657)
key, subkey = jax.random.split(key)
LECs_candidates = sample_LECs(
    subkey,
    500,
    potential.LECs,
    scale_min=0.0,
    scale_max=2.0,
    static_indices=static_indices,
)


emulator.fit(LECs_candidates)

# prior predictive?

prior_scale = 0.2
likelihood_scale = 0.1

sampler = Sampler(
    emulator, sigmas,
    prior_scale=prior_scale,
    likelihood_scale=likelihood_scale,
    static_indices=static_indices
)

n_chains = 20
n_equil = 10
n_skip = 10
n_samples_per_chain = 500
init_noise = 0.5
step_scale = 0.5

start_time = time.time()


        
LECs_em, sigmas_em, err_sigmas_em, params_c_em, params_cc_em, accept_rate_em = sampler.sample(
    n_chains=n_chains,
    n_equil=n_equil,
    n_skip=n_skip,
    n_samples_per_chain=n_samples_per_chain,
    init_noise=init_noise,
    step_scale=step_scale
)

print("Emulator acceptance rate = ", accept_rate_em)
print("Time = ", time.time() - start_time)

filename = "saved_samples/test_emulator_samples_0.2_0.1_n2.npz"
kwargs = {
    "LECs_em": LECs_em,
    "sigmas_em": sigmas_em,
    "err_sigmas_em": err_sigmas_em,
    "params_c_em": params_c_em,
    "params_cc_em": params_cc_em,
    "best fit LECs": potential.LECs,
    "MAP LECs": sampler.MAP_LECs
}
jnp.savez(filename, **kwargs)



start_time = time.time()


LECs_ex, sigmas_ex, err_sigmas_ex, params_c_ex, params_cc_ex, accept_rate_ex = sampler.sample(
    n_chains=n_chains,
    n_equil=n_equil,
    n_skip=n_skip,
    n_samples_per_chain=n_samples_per_chain,
    init_noise=init_noise,
    step_scale=step_scale,
    use_emulator=False,
    seed=317
)

#LECs_ex, sigmas_ex, err_sigmas_ex, params_c_ex, params_cc_ex, accept_rate_ex = LECs_em, sigmas_em, err_sigmas_em, params_c_em, params_cc_em, accept_rate_em

print("Solver acceptance rate = ", accept_rate_ex)
print("Time = ", time.time() - start_time)

filename = "saved_samples/test_solver_samples_0.2_0.1_n2.npz"
kwargs = {
    "LECs_ex": LECs_ex,
    "sigmas_ex": sigmas_ex,
    "err_sigmas_ex": err_sigmas_ex,
    "params_c_ex": params_c_ex,
    "params_cc_ex": params_cc_ex,
    "best fit LECs": potential.LECs,
    "MAP LECs": sampler.MAP_LECs
}
jnp.savez(filename, **kwargs)


