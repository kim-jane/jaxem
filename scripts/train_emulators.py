import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
from jaxem import *
from collections import defaultdict
import corner



# data
filename = "data/np_SGT/PWA93_1_300.txt"
Elabs, sigmas = np.loadtxt(filename, unpack=True)
indices = [0,1,2,5,10,20,50,100,200,300]
Elabs = Elabs[indices]
sigmas = sigmas[indices]



#############

n_mesh = 120
Jmax = 4
static_indices = [0, 10, 11]

mesh = TRNS(n_mesh=n_mesh)
channels = Channels(isospin_channel='np', Jmax=Jmax)
potential = Chiral()
solver = Solver(mesh, channels, potential, Elabs)
emulator = Emulator(solver)


# train emulator
key = jax.random.PRNGKey(317)
key, subkey = jax.random.split(key)
LECs_candidates = sample_LECs(
    subkey,
    500,
    potential.LECs,
    scale_min=0.0,
    scale_max=2.0,
    static_indices=[0, 10, 11]
)


emulator.fit(LECs_candidates)



sampler = Sampler(
    emulator, sigmas,
    prior_scale = 0.5,
    likelihood_scale=0.1,
    static_indices=static_indices
)



'''
times = []

for i in range(100):

    t = time.time()
    sigma_tot = solver.total_cross_sections(
        emulator.t(potential.LECs)
    )
    times.append(time.time() - t)
    
times = jnp.array(times[2:])
print(jnp.mean(times))
print(jnp.std(times))

times = []

for i in range(100):

    t = time.time()
    sigma_tot = solver.total_cross_sections(
        solver.t(potential.LECs)
    )
    times.append(time.time() - t)
    
times = jnp.array(times[2:])
print(jnp.mean(times))
print(jnp.std(times))





sigma_tot = solver.total_cross_sections(t)

plt.plot(solver.Elabs, sigma_tot)

'''
