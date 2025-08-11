import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jaxem import *
import matplotlib.pyplot as plt
import time
from collections import defaultdict


Elab = 10.0
n_mesh = 40
Jmax = 2

Elabs = jnp.linspace(0.001, 300.0, 150)

mesh = TRNS(n_mesh=n_mesh)
channels = Channels(isospin_channel='np', Jmax=Jmax)
potential = Chiral()
solver = Solver(mesh, channels, potential)


for channel in channels.coupled.keys():

    # Christian's phase shifts
    Elabs, delta_singlet, delta_triplet, epsilon_CD = np.loadtxt(f'test/benchmark/N2LO_fulllocal_R0_1.0_lam_50.0_Np_54/phaseShifts_{channel}np.txt', unpack=True)
    
    fig, ax = plt.subplots(1, 3, figsize=(12,4))

    
    ax[0].plot(Elabs, delta_singlet, color='C0')
    ax[1].plot(Elabs, delta_triplet, color='C0')
    ax[2].plot(Elabs, epsilon_CD, color='C0')
    
    for Elab in Elabs[::10]:
    
        Tq = solver.t(channel, Elab)
        observables = solver.calc_observables(channel, Elab, Tq)
        (delta_minus, delta_plus, epsilon), (eta_minus, eta_plus), sigma = observables
        
        '''
        plt.plot(mesh.q, Tq[0,0,1:].real)
        plt.plot(mesh.q, Tq[0,0,1:].imag)
        plt.plot(mesh.q, Tq[0,1,1:].real)
        plt.plot(mesh.q, Tq[0,1,1:].imag)
        plt.plot(mesh.q, Tq[1,0,1:].real)
        plt.plot(mesh.q, Tq[1,0,1:].imag)
        plt.plot(mesh.q, Tq[1,1,1:].real)
        plt.plot(mesh.q, Tq[1,1,1:].imag)
        plt.show()
        '''
        
        ax[0].scatter(Elab, (180/jnp.pi) * delta_minus, color='C1', linestyle='dashed')
        ax[1].scatter(Elab, (180/jnp.pi) * delta_plus, color='C1', linestyle='dashed')
        ax[2].scatter(Elab, (180/jnp.pi) * epsilon, color='C1', linestyle='dashed')
    
    plt.show()




for channel in channels.single.keys():
    
    
    Elab, delta = np.loadtxt(f'test/benchmark/N2LO_fulllocal_R0_1.0_lam_50.0_Np_54/phaseShifts_{channel}np.txt', unpack=True)
    
    plt.plot(Elab, delta)
    
    plt.show()


