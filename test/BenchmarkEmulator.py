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
config = Config('test/BenchmarkEmulator.ini')


# create tmat
tmat = TMatrix(config)
LECs_best = tmat.pot.LECs

#Tsckq, Tscckq = tmat.solve(LECs_best)

#Tq = tmat.solve_single(LECs_best, 0, 0)

Tq = tmat.solve_coupled(LECs_best, 0, 0)


LECs_lbd = 0.0 * LECs_best
LECs_lbd = LECs_lbd.at[0].set(LECs_best[0])

LECs_ubd = 2.0 * LECs_best
LECs_ubd = LECs_ubd.at[0].set(LECs_best[0])


#Tsq = tmat.train_POD_GROM_single(LECs_lbd, LECs_ubd)

Tsq = tmat.train_POD_GROM_coupled(LECs_lbd, LECs_ubd, 0, 0)
print("emulated = ", Tsq.shape)

for i in range(2):
    for j in range(2):
    

        plt.plot(Tq[i,j], 'o-', color='b')

        plt.plot(Tsq[0,i,j,:], 'o:', color='r')
        
        plt.show()
        plt.close()



'''
for cc in range(tmat.chan.Ncoupled):

    for k in range(tmat.Nk):

        fig, ax = plt.subplots(2, 1, figsize=(6, 8), dpi=300)
        
        ms = 4
        
        ax[0].plot(tmat.q, Tscckq[0,cc,k,0,0,1:].real, marker='o', markersize=ms)
        ax[0].plot(tmat.q, Tscckq[0,cc,k,1,1,1:].real, marker='o', markersize=ms)
        ax[0].plot(tmat.q, Tscckq[0,cc,k,0,1,1:].real, marker='o', markersize=ms)
        ax[0].plot(tmat.q, Tscckq[0,cc,k,1,0,1:].real, marker='o', markersize=ms)
        
        ax[1].plot(tmat.q, Tscckq[0,cc,k,0,0,1:].imag, marker='o', markersize=ms)
        ax[1].plot(tmat.q, Tscckq[0,cc,k,1,1,1:].imag, marker='o', markersize=ms)
        ax[1].plot(tmat.q, Tscckq[0,cc,k,0,1,1:].imag, marker='o', markersize=ms)
        ax[1].plot(tmat.q, Tscckq[0,cc,k,1,0,1:].imag, marker='o', markersize=ms)

        ax[0].axvline(config.p1, color='k', linestyle='dashed')
        ax[0].axvline(config.p2, color='k', linestyle='dashed')
        ax[0].axvline(config.p3, color='k', linestyle='dashed')
        ax[0].axvline(tmat.k[k], color='r', linestyle='dashed')

        ax[1].axvline(config.p1, color='k', linestyle='dashed')
        ax[1].axvline(config.p2, color='k', linestyle='dashed')
        ax[1].axvline(config.p3, color='k', linestyle='dashed')
        ax[1].axvline(tmat.k[k], color='r', linestyle='dashed')
        
        
        plt.savefig(f'figures/{config.output}_tmat_{tmat.chan.coupled_spect_not[cc]}_np_Elab{tmat.Elab[k]:08.3f}.pdf', format='pdf')
        plt.close()



# compute phase shifts
single_output, coupled_output = tmat.phase_shifts(T_single=Tsckq, T_coupled=Tscckq)


# plot

if single_output is not None:

    delta_sck, eta_sck = single_output
    
    for c in range(tmat.chan.Nsingle):
    
        plt.figure(figsize=(6,4), dpi=300)
    
        # Christian's phase shifts
        Elab, delta = np.loadtxt(f'test/benchmark/N2LO_fulllocal_R0_1.0_lam_50.0_Np_54/phaseShifts_{tmat.chan.single_spect_not[c]}np.txt', unpack=True)
        plt.plot(Elab, delta)
        
        # my phase shifts
        plt.scatter(tmat.Elab, (180/jnp.pi) * delta_sck[0,c,:])
        
        plt.savefig(f'figures/{config.output}_phaseshift_{tmat.chan.single_spect_not[c]}_np.pdf', format='pdf')
        plt.close()
        
        

if coupled_output is not None:

    delta_minus_scck, delta_plus_scck, epsilon_scck, eta_minus_scck, eta_plus_scck = coupled_output

    for cc in range(tmat.chan.Ncoupled):
    
        fig, ax = plt.subplots(1, 3, figsize=(12,4), dpi=300)
        
        # Christian's phase shifts
        Elab, delta_singlet, delta_triplet, epsilon = np.loadtxt(f'test/benchmark/N2LO_fulllocal_R0_1.0_lam_50.0_Np_54/phaseShifts_{tmat.chan.coupled_spect_not[cc]}np.txt', unpack=True)
        ax[0].plot(Elab, delta_singlet, color='C0')
        ax[1].plot(Elab, delta_triplet, color='C0')
        ax[2].plot(Elab, epsilon, color='C0')
        
        # my phase shifts
        ax[0].scatter(tmat.Elab, (180/jnp.pi) * delta_minus_scck[0,cc,:], color='C1')
        ax[1].scatter(tmat.Elab, (180/jnp.pi) * delta_plus_scck[0,cc,:], color='C1')
        ax[2].scatter(tmat.Elab, (180/jnp.pi) * epsilon_scck[0,cc,:], color='C1')
        
        plt.savefig(f'figures/{config.output}_phaseshift_{tmat.chan.coupled_spect_not[cc]}_np.pdf', format='pdf')
        plt.close()



'''
