import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
from jaxem import *
from collections import defaultdict
import corner


# experimental data
filename = "test/benchmark/np_SGT/PWA93_1_300.txt"
Elabs_data, sigmas_data = np.loadtxt(filename, unpack=True)

# plot cross sections
plt.figure(figsize=(6,5), dpi=300)

plt.plot(Elabs_data, sigmas_data, color='k', label='Experiment', linestyle='dashed')



mesh = TRNS(n_mesh=40)
channels = Channels(isospin_channel='np', Jmax=2)
potential = Chiral()
solver = Solver(mesh, channels, potential)
emulator = Emulator(solver)


# load samples from file
filename = f"saved_samples/chiral_Nq{mesh.n_mesh}_Jmax{channels.Jmax}_eta2.0.npz"
data = jnp.load(filename)
LECs_samples = data["LECs_samples"]
sigmas_samples = data["sigmas_samples"]
Elabs_train = data["Elabs"]
sigmas_train = data["sigmas_data"]
    

# make the corner plot
fig = corner.corner(
    np.array(LECs_samples[:,1:10]),
    truths=potential.LECs[1:10],
    labels=[r"$C_S$", r"$C_T$", "$C_1$", "$C_2$", "$C_3$", "$C_4$", "$C_5$", "$C_6$", "$C_7$"])

plt.savefig(f"figures/chiral_corner_eta2.0.pdf", format="pdf")
plt.clf()

