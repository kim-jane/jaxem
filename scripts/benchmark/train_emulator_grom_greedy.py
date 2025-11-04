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
    static_indices=[0,10,11],
)
print(LECs_candidates.shape)

emulator.fit(LECs_candidates, rom='g', mode='greedy')
