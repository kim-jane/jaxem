"""
This script solves the Lippmann-Schwinger 
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
from jaxem import *
from collections import defaultdict
import corner



n_mesh = 40
Jmax = 2
Elabs = jnp.linspace(0.001, 300.0, 10)
Elabs = [0.1, 1, 2, 5, 10, 20, 50, 100, 200, 300]

mesh = TRNS(n_mesh=n_mesh)
channels = Channels(isospin_channel='np', Jmax=Jmax)
potential = Chiral()
solver = Solver(mesh, channels, potential, Elabs)



t_and_err_onshell = solver.onshell_t_and_err(potential.LECs)
sigma_tot, err_sigma_tot, single_output, coupled_output = solver.scattering_params(t_and_err_onshell)
delta_ck, eta_ck, sigma_ck = single_output
delta_minus, delta_plus, epsilon, eta_minus, eta_plus, sigma_cck = coupled_output

directory = "benchmark/N2LO_fulllocal_R0_1.0_lam_50.0_Np_54/"
'''
for cc, label in enumerate(channels.coupled_labels):
    filename = directory + f"phaseShifts_{label}np.txt"
    E_CD, delta_m_CD, delta_p_CD, eps_CD = np.loadtxt(filename, unpack=True)
    
    fig, axes = plt.subplots(1, 3)
    axes[0].plot(E_CD, delta_m_CD)
    axes[1].plot(E_CD, delta_p_CD)
    axes[2].plot(E_CD, eps_CD)
    
    axes[0].scatter(Elabs, delta_minus[cc] * (180./jnp.pi))
    axes[1].scatter(Elabs, delta_plus[cc] * (180./jnp.pi))
    axes[2].scatter(Elabs, epsilon[cc] * (180./jnp.pi))
    plt.show()
    plt.close()

    print(label)
    
    
for c, label in enumerate(channels.single_labels):
    filename = directory + f"phaseShifts_{label}np.txt"
    E_CD, delta_CD = np.loadtxt(filename, unpack=True)
    
    plt.plot(E_CD, delta_CD)
    plt.scatter(Elabs, delta_ck[c] * (180./jnp.pi))
    plt.show()
    plt.close()
    print(label)

'''

# data
filename = "benchmark/np_SGT/PWA93_0.1_300.txt"
Elabs_PWA, sigmas_PWA = np.loadtxt(filename, unpack=True)

plt.plot(Elabs_PWA, sigmas_PWA)
plt.plot(Elabs, sigma_tot)
plt.show()

for cc, label in enumerate(channels.coupled_labels):
    print(label)
    plt.scatter(Elabs, sigma_cck[cc])
    plt.show()
    plt.close()

    
    
    
for c, label in enumerate(channels.single_labels):
    print(label)
    plt.scatter(Elabs, sigma_ck[c])
    plt.show()
    plt.close()
