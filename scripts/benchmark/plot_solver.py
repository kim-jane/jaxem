import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import time
from jaxem import *
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import plot_config


# instantiate objects
config = Config('scripts/benchmark/config.ini')
mesh = TRNS(n_mesh=config.n_mesh)
channels = Channels(isospin_channel=config.isospin_channel, Jmax=config.Jmax)
potential = Chiral()
solver = Solver(mesh, channels, potential, config.Elabs, setup=False)


# load results from output file
data = jnp.load('scripts/benchmark/solver_output.npz')

sigma_tot = data['sigma_tot']
single_output = data['single_output']
coupled_output = data['coupled_output']
print(sigma_tot.shape)
print(single_output.shape)
print(coupled_output.shape)


###################################
###     TOTAL CROSS SECTION     ###
###################################

fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))

# PWA93 benchmark
filename = "benchmark/np_SGT/PWA93_0.1_300.txt"
Elabs_PWA, sigmas_PWA = np.loadtxt(filename, unpack=True)
ax.plot(Elabs_PWA, sigmas_PWA, color='k', label='PWA93', zorder=0)

ax.scatter(config.Elabs, sigma_tot, color='C0', marker='o', label='High-fidelity', zorder=1)

ax.set_yscale("log")
ax.set_ylabel("$\sigma_{tot}$ [mb]")
ax.set_xlabel("$E_{lab}$ [MeV]")
ax.set_xlim(0, 300)
plt.legend()
#plt.show()
plt.savefig("scripts/benchmark/figures/solver_total_cross_section.pdf", format="pdf")
plt.close()



###########################################
###     SINGLE CHANNEL PHASE SHIFTS     ###
###########################################

def spectro_label(s):
    return fr"$^{{{s[0]}}}{s[1]}_{{{s[2:]}}}$"
    
f = 180. / jnp.pi

# unpack single channel outputs
delta, eta, sigma = single_output

for c, label in enumerate(channels.single_labels):
    
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))
    
    # plot PWA phase shifts if they exist
    filename_PWA = f"benchmark/np_phase_shifts/PWA93_0.1_300_{label}.txt"
    
    try:
        Elabs_PWA, delta_PWA = np.loadtxt(filename_PWA, unpack=True)
        ax.plot(Elabs_PWA, delta_PWA, color='k', label='PWA93')
        
    except FileNotFoundError:
        print(f"PWA93 benchmark file for {label} not found, skipping.")
        continue
    
    # plot Christian Drischler's phase shifts if they exist
    filename_CD = f"benchmark/N2LO_fulllocal_R0_1.0_lam_50.0_Np_54/phaseShifts_{label}np.txt"
    
    try:
        Elabs_CD, delta_CD = np.loadtxt(filename_CD, unpack=True)
        ax.plot(Elabs_CD, delta_CD, color='C3', linestyle='dashed', label='CD')
        
    except FileNotFoundError:
        print(f"CD benchmark file for {label} not found, skipping.")
        continue
        
    ax.scatter(config.Elabs, f * delta[c], color='C0', marker='o', label='High-fidelity', zorder=10)
        
    ax.set_ylabel("Phase Shift [deg]")
    ax.set_xlabel("$E_{lab}$ [MeV]")
    ax.set_xlim(0, 300)
    plt.text(
        0.95, 0.95,
        spectro_label(label),
        transform=plt.gca().transAxes,  # coordinates relative to axes
        fontsize=14,
        ha='right', va='top',     # horizontal and vertical alignment
        color='k',
        bbox=dict(
            boxstyle='round,pad=0.3',
            facecolor='white',
            edgecolor='black'
        ),
        zorder=100
    )
    plt.legend()
    #plt.show()
    plt.savefig(f"scripts/benchmark/figures/solver_phase_shift_{label}.pdf", format="pdf")
    plt.close()


############################################
###     COUPLED CHANNELS PHASE SHIFTS    ###
############################################

# unpack coupled channel outputs
delta_minus, delta_plus, epsilon, eta_minus, eta_plus, sigma = coupled_output



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
