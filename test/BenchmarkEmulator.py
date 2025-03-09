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

from matplotlib import rc
rc('font',**{'family':'serif'})
rc('text', usetex=True)

figkwargs = {
    'num': None,
    'dpi': 400,
    'facecolor': 'w',
    'edgecolor': 'k',
}

fontsize = 18
labelsize = 16
markersize = 10
linewidth = 3
gridalpha = 0.2


ti = time.time()


n_devices = jax.local_device_count()
print(f"n_devices = {n_devices}")


# read inputs
config = Config('test/BenchmarkEmulator.ini')


# create tmat
tmat = TMatrix(config)


POD_training = tmat.train_POD_GROM()
greedy_training = tmat.train_greedy_GROM()

POD_obs = tmat.observables(POD_training)
greedy_obs = tmat.observables(greedy_training)
print(POD_obs)

# exact calculations
Tkq_exact = jax.vmap(tmat.solve_coupled, in_axes=(None,None,0))(tmat.pot.LECs, 0, jnp.arange(tmat.Nk))

Tsckq, Tscckq = tmat.solve(tmat.pot.LECs)
single_output, coupled_output = tmat.phase_shifts_old(T_single=Tsckq, T_coupled=Tscckq)
delta_minus_scck, delta_plus_scck, epsilon_scck, eta_minus_scck, eta_plus_scck = coupled_output

print("***", delta_minus_scck.shape)





'''
delta_exact = (delta_minus_scck[0,0], delta_plus_scck[0,0], epsilon_scck[0,0])


conv = 180. / jnp.pi

fig, axes = plt.subplots(2, 3, figsize=(14,6), gridspec_kw={'height_ratios': [2, 1]}, **figkwargs)

energies = tmat.Elab

delta_minus_exact = conv * delta_plus_scck[0,0]
delta_minus_POD = conv * POD_obs['3S1-3D1']['phase shift'][:,1]
delta_minus_greedy = conv * greedy_obs['3S1-3D1']['phase shift'][:,1]
delta_minus_POD_error = np.abs(delta_minus_exact-delta_minus_POD)
delta_minus_greedy_error = np.abs(delta_minus_exact-delta_minus_greedy)


delta_plus_exact = conv * delta_minus_scck[0,0]
delta_plus_POD = conv * POD_obs['3S1-3D1']['phase shift'][:,0]
delta_plus_greedy = conv * POD_obs['3S1-3D1']['phase shift'][:,0]
delta_plus_POD_error = np.abs(delta_plus_exact-delta_plus_POD)
delta_plus_greedy_error = np.abs(delta_plus_exact-delta_plus_greedy)


epsilon_exact = conv * epsilon_scck[0,0]
epsilon_POD = conv * POD_obs['3S1-3D1']['phase shift'][:,2]
epsilon_POD_error = np.abs(epsilon_exact-epsilon_POD)
epsilon_greedy = conv * greedy_obs['3S1-3D1']['phase shift'][:,2]
epsilon_greedy_error = np.abs(epsilon_exact-epsilon_greedy)

lw = 3
ms = 7

POD_kwargs = {'label' : 'POD', 'linestyle' : 'dashed', 'color' : 'C0', 'marker' : 'o', 'markersize' : ms, 'linewidth': lw}
greedy_kwargs = {'label' : 'Greedy', 'linestyle' : 'dotted', 'color' : 'C1', 'marker' : '^', 'markersize' : ms}

axes[0,0].plot(energies, delta_plus_exact, linewidth=lw, color='k')
axes[0,0].plot(energies, delta_plus_POD, **POD_kwargs)
axes[0,0].plot(energies, delta_plus_greedy, **greedy_kwargs)
axes[1,0].plot(energies, delta_plus_POD_error, **POD_kwargs)
axes[1,0].plot(energies, delta_plus_greedy_error, **greedy_kwargs)

axes[0,1].plot(energies, delta_minus_exact, linewidth=lw, color='k')
axes[0,1].plot(energies, delta_minus_POD, **POD_kwargs)
axes[0,1].plot(energies, delta_minus_greedy, **greedy_kwargs)
axes[1,1].plot(energies, delta_minus_POD_error, **POD_kwargs)
axes[1,1].plot(energies, delta_minus_greedy_error, **greedy_kwargs)

axes[0,2].plot(energies, epsilon_exact, linewidth=lw, label='Exact', color='k')
axes[0,2].plot(energies, epsilon_POD, **POD_kwargs)
axes[0,2].plot(energies, epsilon_greedy, **greedy_kwargs)
axes[1,2].plot(energies, epsilon_POD_error, **POD_kwargs)
axes[1,2].plot(energies, epsilon_greedy_error, **greedy_kwargs)

axes[1,0].set_yscale('log')
axes[1,1].set_yscale('log')
axes[1,2].set_yscale('log')


axes[0,0].set_xticklabels([])
axes[0,1].set_xticklabels([])
axes[0,2].set_xticklabels([])

axes[1,0].set_xlabel(r'$E_{lab}$ [MeV]', fontsize=fontsize)
axes[1,1].set_xlabel(r'$E_{lab}$ [MeV]', fontsize=fontsize)
axes[1,2].set_xlabel(r'$E_{lab}$ [MeV]', fontsize=fontsize)

axes[0,0].set_ylabel(r'Phase Shift [degrees]', fontsize=fontsize)
axes[0,2].set_ylabel(r'Mixing Angle [degrees]', fontsize=fontsize)
axes[1,0].set_ylabel(r'Abs Error [degrees]', fontsize=fontsize)

axes[0,0].text(250, 150, r'$^3S_1$', fontsize=24, color='k')
axes[0,1].text(250, -5, r'$^3D_1$', fontsize=24, color='k')
axes[0,2].text(260, 2.35, r'$\varepsilon_1$', fontsize=24, color='k')


for ax in axes.ravel():
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 300)
    ax.tick_params(axis='both', labelsize=labelsize)


axes[0,2].legend(fontsize=fontsize, loc='lower center')
plt.tight_layout()


# reduce space between the first and second columns
axes[0, 1].set_position([axes[0, 1].get_position().x0 - 0.01, axes[0, 1].get_position().y0,
                        axes[0, 1].get_position().width, axes[0, 1].get_position().height])
axes[1, 1].set_position([axes[1, 1].get_position().x0 - 0.01, axes[1, 1].get_position().y0,
                        axes[1, 1].get_position().width, axes[1, 1].get_position().height])

plt.savefig("phase_shifts_3S1-3D1.pdf", format="pdf")
plt.close()


for chan in tmat.chan.single_spect_not:
    for k in range(tmat.Nk):
        plt.plot(POD_training[chan][k]['Tq at best LECs'][-1])
        print(chan, tmat.Elab[k], POD_training[chan][k]['max cal error'])
        
        #plt.plot(greedy_training[chan][k]['Tq at best LECs'][-1], linestyle='dashed')
        #print(chan, tmat.Elab[k], greedy_training[chan][k]['max cal error'][-1])
        
        plt.show()
        plt.close()


for chan in tmat.chan.coupled_spect_not:
    for k in range(tmat.Nk):
    
        plt.plot(tmat.q, POD_training[chan][k]['Tq at best LECs'][-1][0,0,1:])
        plt.plot(tmat.q, POD_training[chan][k]['Tq at best LECs'][-1][0,1,1:])
        plt.plot(tmat.q, POD_training[chan][k]['Tq at best LECs'][-1][1,0,1:])
        plt.plot(tmat.q, POD_training[chan][k]['Tq at best LECs'][-1][1,1,1:])
        
        plt.scatter(tmat.k[k], POD_training[chan][k]['Tq at best LECs'][-1][0,0,0])
        plt.scatter(tmat.k[k], POD_training[chan][k]['Tq at best LECs'][-1][0,1,0])
        plt.scatter(tmat.k[k], POD_training[chan][k]['Tq at best LECs'][-1][1,0,0])
        plt.scatter(tmat.k[k], POD_training[chan][k]['Tq at best LECs'][-1][1,1,0])
        print(chan, tmat.Elab[k], POD_training[chan][k]['max cal error'])
        
        
        plt.plot(greedy_training[chan][k]['Tq at best LECs'][-1][0,0], linestyle='dashed')
        plt.plot(greedy_training[chan][k]['Tq at best LECs'][-1][0,1], linestyle='dashed')
        plt.plot(greedy_training[chan][k]['Tq at best LECs'][-1][1,0], linestyle='dashed')
        plt.plot(greedy_training[chan][k]['Tq at best LECs'][-1][1,1], linestyle='dashed')
        print(chan, tmat.Elab[k], greedy_training[chan][k]['max cal error'][-1])
        
        plt.show()
        plt.close()

'''
