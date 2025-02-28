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
config = Config('test/BenchmarkMalflietTjon.ini')

# create tmat
tmat = TMatrix(config)

# best fit LECs
LECs_best = tmat.pot.LECs

T, _ = tmat.solve(LECs_best)
T = T[0]

print("min T = ", jnp.min(T))
print("max T = ", jnp.max(T))

T *= config.hbarc # MeV fm^3


# plot T matrix against benchmark
fig, axes = plt.subplots(2, 2, sharex=True, figsize=(8,6), dpi=300)

colors = ['C0', 'C2', 'C1', 'C3']

for j in range(4):
    axes[1,1].plot([-1], [0], c=colors[j], lw=3, label=r'$E_{lab}=$ '+str(round(tmat.Elab[j]))+' MeV')
    
axes[1,1].plot([-1], [0], 'k:', lw=2, label='Benchmark')
axes[1,1].legend(fontsize=12, ncol=1, loc='lower right')


data = np.loadtxt('test/benchmark/Elster/half-shell-t-MT3.dat', skiprows=1, usecols=(2,3,4,5))
data = np.reshape(data, (2, 4, 33, 4))


# plot curves
for i in range(2):
    for j in range(4):
    
        axes[0,i].plot(tmat.q, T[i,j,1:].real, lw=3, c=colors[j])
        axes[1,i].plot(tmat.q, T[i,j,1:].imag, lw=3, c=colors[j])

        #axes[0,i].set_ylim(-100, 10)
        q = data[i,j,:-1,0]
        q0 = data[i,j,0,1]
        ReT = data[i,j,:-1,2]
        ImT = data[i,j,:-1,3]

        axes[0,i].plot(q, ReT, 'k:', lw=2)
        axes[1,i].plot(q, ImT, 'k:', lw=2)


axes[0,0].set_title(r'$\ell=0$', fontsize=12)
axes[0,1].set_title(r'$\ell=1$', fontsize=12)
axes[0,0].set_ylabel(r'$\mathcal{Re} \ T_\ell(k, k_0; k_0)$  [MeV fm$^3$]', fontsize=12)
axes[1,0].set_ylabel(r'$\mathcal{Im} \ T_\ell(k, k_0; k_0)$  [MeV fm$^3$]', fontsize=12)
axes[1,0].set_xlabel(r'$k$ [fm$^{-1}$]', fontsize=12)
axes[1,1].set_xlabel(r'$k$ [fm$^{-1}$]', fontsize=12)


axes[0,0].yaxis.set_major_locator(MaxNLocator(integer=True))
axes[0,1].yaxis.set_major_locator(MaxNLocator(integer=True))
axes[1,0].yaxis.set_major_locator(MaxNLocator(integer=True))
axes[1,1].yaxis.set_major_locator(MaxNLocator(integer=True))

axes[0,0].grid(alpha=0.4)
axes[1,0].grid(alpha=0.4)
axes[0,1].grid(alpha=0.4)
axes[1,1].grid(alpha=0.4)

#axes[0,0].set_ylim(-20, 6)
#axes[0,1].set_ylim(-18, 3)
#axes[1,0].set_ylim(-62, 16)
#axes[1,1].set_ylim(-9, 2)
axes[1,0].set_xlim(0, 16)

plt.tight_layout()
plt.savefig(f'figures/{config.output}_tmat.pdf', format='pdf')
plt.close()

