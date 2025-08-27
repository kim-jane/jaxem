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


for j in range(1,5):

    mesh = TRNS(n_mesh=40)
    channels = Channels(isospin_channel='np', Jmax=j)
    potential = Chiral()
    solver = Solver(mesh, channels, potential)
    emulator = Emulator(solver)


    # load samples from file
    filename = f"saved_samples/chiral_Nq{mesh.n_mesh}_Jmax{channels.Jmax}.npz"
    data = jnp.load(filename)
    LECs_samples = data["LECs_samples"]
    sigmas_samples = data["sigmas_samples"]
    Elabs_train = data["Elabs"]
    sigmas_train = data["sigmas_data"]
    
    if j == 1:
        plt.plot(Elabs_train, sigmas_train, color='k', marker='o', label='Training Points', linewidth=0.0)


    sigmas_avg = jnp.mean(sigmas_samples, axis=0) * 10.
    sigmas_std = jnp.std(sigmas_samples, axis=0) * 10.

    print(sigmas_avg)
    print(sigmas_std)

    

    plt.errorbar(
        Elabs_train, sigmas_avg,
        yerr=3*sigmas_std,
        marker='^', linewidth=0.0,
        elinewidth=1.0, capsize=5,
        label=r'$J_{max}=$'+str(j),
        color='C'+str(j-1)
    )
    
    plt.fill_between(
        Elabs_train,
        sigmas_avg - 3 * sigmas_std,
        sigmas_avg + 3 * sigmas_std,
        alpha=0.5, color='C'+str(j-1)
    )
    
plt.yscale('log')
plt.xscale('log')
plt.xlim(0.07, 500)
plt.legend()
plt.xlabel(r"$E_{lab}$ [MeV]")
plt.ylabel(r"$\sigma_{tot}$ [mb]")
plt.savefig(f"figures/chiral_cross_section_samples_0_300_loglog.pdf", format="pdf")
#plt.clf()
plt.show()

