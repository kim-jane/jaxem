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

###

'''
pot = Potential(config)
LECs_best = pot.LECs

p1, p2, Vpp = np.loadtxt('test/benchmark/VNN_N2LO_local_R0_1.0_SLLJT_00001_lambda_50.00_Np_200_np.dat', unpack=True)

p1 = jnp.array(p1)
p2 = jnp.array(p2)
Vpp = jnp.reshape(jnp.array(Vpp), (200, 200))
print(Vpp.shape)

p1 = p1[::200]
p2 = p2[:200]


Vocp_test, _ = pot.Voc(config.hbarc * p1, diag=True)
Vp_test = jnp.einsum('o,oci->ci', LECs_best, Vocp_test)[0]

plt.plot(jnp.diag(Vpp)/config.m/config.hbarc )
plt.plot(Vp_test)
plt.show()
'''

###


Elab, delta = np.loadtxt('test/benchmark/N2LO_fulllocal_R0_1.0_lam_50.0_Np_54/phaseShifts_1S0np.txt', unpack=True)

plt.plot(Elab, delta)

Elab, delta = np.loadtxt('test/benchmark/N2LO_fulllocal_R0_1.0_lam_50.0_Np_54/phaseShifts_3P0np.txt', unpack=True)

plt.plot(Elab, delta)

'''
plt.show()


pot = Potential(config)
LECs_best = pot.LECs

p1, p2, Vpp = np.loadtxt('test/benchmark/VNN_N2LO_local_R0_1.0_SLLJT_00001_lambda_50.00_Np_200_np.dat', unpack=True)

p1 = jnp.array(p1)
p2 = jnp.array(p2)
Vpp = jnp.reshape(jnp.array(Vpp), (200, 200))
print(Vpp.shape)

p1 = p1[::200]
p2 = p2[:200]


Vocp_test, _ = pot.Voc(config.hbarc * p1, diag=True)
Vp_test = jnp.einsum('o,oci->ci', LECs_best, Vocp_test)[0]

plt.plot(jnp.diag(Vpp)/config.m/config.hbarc )
plt.plot(Vp_test)
plt.show()

'''





# create tmat
tmat = TMatrix(config)
LECs_best = tmat.pot.LECs


Tsckq, Tscckq = tmat.solve(LECs_best)
print("Tsckq = ", Tsckq.shape)
print(tmat.k[1])
print(Tsckq[0,0,1,0])

delta, _ = tmat.phase_shifts(T_single=Tsckq, T_coupled=Tscckq)
delta = delta[0]
print("delta = ", delta)

for i in range(delta.shape[0]):
    plt.scatter(tmat.Elab, (180./np.pi) * delta[i])
    

Elab_AG = jnp.array([20., 40., 60., 80., 100.])
delta_AG_singlet = jnp.array([9.339387E-01, 7.701090E-01, 6.477976E-01, 5.488205E-01, 4.653679E-01])
delta_AG_triplet = jnp.array([1.222418E-01, 1.811798E-01, 1.919273E-01, 1.782547E-01, 1.519650E-01])
plt.scatter(Elab_AG, (180./np.pi) * delta_AG_singlet, marker='s')
plt.scatter(Elab_AG, (180./np.pi) * delta_AG_triplet, marker='s')
    
plt.show()
