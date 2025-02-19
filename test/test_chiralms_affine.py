import sys
sys.path.append("./chiral/modules")

import numpy as np
import matplotlib.pyplot as plt
from Potential import Potential, chiralms, chiralms_affine
from Channel import Channel
import jax
import jax.numpy as jnp

potentialArgs = {"label": "chiral", "kwargs": {"potId": 213}}
print(potentialArgs)
channel = Channel(S=0, L=0, LL=0, J=0, channel=0)
#potential = Potential(channel, **potentialArgs)

def V_chiral(k, kk):
    return chiralms(k, kk, channel, use_default_lec_values=True, **potentialArgs["kwargs"])
    
    
def V_chiral_affine(k, kk):
    return chiralms_affine(k, kk, channel, **potentialArgs["kwargs"])


CS = 5.43850     # fm^2
CT = 0.27672     # fm^2
C1 = -0.14084    # fm^4
C2 = 0.04243     # fm^4
C3 = -0.12338    # fm^4
C4 = 0.11018     # fm^4
C5 = -2.11254    # fm^4
C6 = 0.15898     # fm^4
C7 = -0.26994    # fm^4
CNN = 0.04344    # fm^2
CPP = 0.062963   # fm^2
LECs = jnp.array([1.0, CS, CT, C1, C2, C3, C4, C5, C6, C7, CNN, CPP])

# plot affine
num_pts = 50
k_arr = jnp.linspace(0., 10., num_pts)
Vop = jnp.zeros((12, num_pts, num_pts))



for j, kk in enumerate(k_arr):
    for i, k in enumerate(k_arr):
        Vtemp = V_chiral_affine(k, kk)
        for o in range(12):
            Vop = Vop.at[o, i, j].set( Vtemp[o] )
            
            
V_affine = jnp.einsum('o,oij->ij', LECs, Vop)
    
# plot full
Vfull = jnp.zeros((num_pts, num_pts))


for j, kk in enumerate(k_arr):
    for i, k in enumerate(k_arr):
        Vfull = Vfull.at[i, j].set( V_chiral(k, kk))


print("| V from chiralms_affine - V from chiralms | = ", jnp.linalg.norm(V_affine - Vfull))
print("V from chiralms_affine = ", V_affine)
print("V from chiralms = ", Vfull)
