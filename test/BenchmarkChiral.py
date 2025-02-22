import jax
from jax import numpy as jnp
from jax import random, vmap
import pickle
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from jaxem import *
import numpy as np
from scipy.linalg import orth

ti = time.time()


n_devices = jax.local_device_count()
print(f"n_devices = {n_devices}")


# read inputs
config = Config('test/BenchmarkChiral.ini')
config.print_info()


# create tmat
tmat = TMatrix(config)
LECs_best = tmat.pot.LECs




Tsckq, Tscckq = tmat.solve(LECs_best)


for cc in range(tmat.chan.Ncoupled):

    plt.plot(tmat.q/config.hbarc, (config.hbarc)**2 * Tscckq[0,cc,0,0,0,1:].real, marker='o')
    plt.plot(tmat.q/config.hbarc, (config.hbarc)**2 * Tscckq[0,cc,0,1,1,1:].real, marker='o')
    plt.plot(tmat.q/config.hbarc, (config.hbarc)**2 * Tscckq[0,cc,0,0,1,1:].real, marker='o')
    plt.plot(tmat.q/config.hbarc, (config.hbarc)**2 * Tscckq[0,cc,0,1,0,1:].real, marker='o')
    
    plt.plot(tmat.q/config.hbarc, (config.hbarc)**2 * Tscckq[0,cc,0,0,0,1:].imag, linestyle='dashed', marker='o')
    plt.plot(tmat.q/config.hbarc, (config.hbarc)**2 * Tscckq[0,cc,0,1,1,1:].imag, linestyle='dashed', marker='o')
    plt.plot(tmat.q/config.hbarc, (config.hbarc)**2 * Tscckq[0,cc,0,0,1,1:].imag, linestyle='dashed', marker='o')
    plt.plot(tmat.q/config.hbarc, (config.hbarc)**2 * Tscckq[0,cc,0,1,0,1:].imag, linestyle='dashed', marker='o')
    
#plt.xlim(0, 20)
plt.show()
plt.close()


#try negative off diagonal potential

# computes phase shifts and plot
single_output, coupled_output = tmat.phase_shifts(T_single=Tsckq, T_coupled=Tscckq)


# plot Christian's phase shifts for all channels
fig, ax = plt.subplots(1, 2, figsize=(12, 6))


for c in range(tmat.chan.Nsingle):
    

    Elab, delta = np.loadtxt(f'test/benchmark/N2LO_fulllocal_R0_1.0_lam_50.0_Np_54/phaseShifts_{tmat.chan.single_spect_not[c]}np.txt', unpack=True)
    ax[0].plot(Elab, delta)


for cc in range(tmat.chan.Ncoupled):

    Elab, delta_singlet, delta_triplet, epsilon = np.loadtxt(f'test/benchmark/N2LO_fulllocal_R0_1.0_lam_50.0_Np_54/phaseShifts_{tmat.chan.coupled_spect_not[cc]}np.txt', unpack=True)
    ax[1].plot(Elab, delta_singlet, color='C'+str(cc))
    ax[1].plot(Elab, delta_triplet, linestyle='dashed', color='C'+str(cc))
    ax[1].plot(Elab, epsilon, linestyle='dotted', color='C'+str(cc))


if single_output is not None:
    delta_sck, eta_sck = single_output
    
    for c in range(tmat.chan.Nsingle):

        ax[0].scatter(tmat.Elab, (180/jnp.pi) * delta_sck[0,c,:])

if coupled_output is not None:
    delta_minus_scck, delta_plus_scck, epsilon_scck, eta_minus_scck, eta_plus_scck = coupled_output

    for cc in range(tmat.chan.Ncoupled):

        ax[1].scatter(tmat.Elab, (180/jnp.pi) * delta_minus_scck[0,cc,:], color='C'+str(cc+1))
        ax[1].scatter(tmat.Elab, (180/jnp.pi) * delta_plus_scck[0,cc,:], color='C'+str(cc+1), linestyle='dashed')
        ax[1].scatter(tmat.Elab, (180/jnp.pi) * epsilon_scck[0,cc,:], color='C'+str(cc+1), linestyle='dotted')

plt.show()
plt.close()



