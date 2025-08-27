import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
from jaxem import *
from collections import defaultdict



mesh = TRNS(n_mesh=40)
channels = Channels(isospin_channel='np', Jmax=4)
potential = Chiral()
solver = Solver(mesh, channels, potential)


# benchmark theoretical calculations
for pot in ["PWA93", "NijmI", "ESC96"]:
    
    filename = f"test/benchmark/np_SGT/{pot}_1_300.txt"
    Elabs, sigmas = np.loadtxt(filename, unpack=True)
    plt.plot(Elabs[::2], sigmas[::2], label=pot)
    

Elabs = jnp.linspace(0.0, 300.0, 10)
sigmas = jnp.zeros_like(Elabs)

for i, Elab in enumerate(Elabs):
    sigmas = sigmas.at[i].set(solver.total_cross_section(Elab).squeeze())

plt.plot(Elabs, 10. * sigmas, color='k', marker='^')
plt.legend()
plt.show()
