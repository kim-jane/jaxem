import sys
sys.path.append("./chiral/modules")

import numpy as np
import matplotlib.pyplot as plt
from Potential import Potential, chiralms, chiralms_affine
from Channel import Channel
import jax
import jax.numpy as jnp


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
potentialArgs = {"label": "chiral", "kwargs": {"potId": 213}}

channel_mm = Channel(S=1, L=0, LL=0, J=1, channel=0)
channel_pp = Channel(S=1, L=2, LL=2, J=1, channel=0)
channel_mp = Channel(S=1, L=0, LL=2, J=1, channel=0)
channel_pm = Channel(S=1, L=2, LL=0, J=1, channel=0)

def V_chiral(k, kk, channel):
    return chiralms(k, kk, channel, use_default_lec_values=True, **potentialArgs["kwargs"])
    
    
def V_chiral_affine(k, kk, channel):
    return chiralms_affine(k, kk, channel, **potentialArgs["kwargs"])

# momentum grid
num_pts = 100
q_arr = jnp.linspace(0., 10., num_pts)


# get coupled affine potential
Voqq_mm = jnp.zeros((12, num_pts, num_pts))
Voqq_pp = jnp.zeros((12, num_pts, num_pts))
Voqq_mp = jnp.zeros((12, num_pts, num_pts))
Voqq_pm = jnp.zeros((12, num_pts, num_pts))

for i, qi in enumerate(q_arr):
    for j, qj in enumerate(q_arr):
    
        Voqq_mm = Voqq_mm.at[:,i,j].set( V_chiral_affine(qi, qj, channel_mm) )
        Voqq_pp = Voqq_pp.at[:,i,j].set( V_chiral_affine(qi, qj, channel_pp) )
        Voqq_mp = Voqq_mp.at[:,i,j].set( V_chiral_affine(qi, qj, channel_mp) )
        Voqq_pm = Voqq_pm.at[:,i,j].set( V_chiral_affine(qi, qj, channel_pm) )
            
            
# check symmetries of each operator
for o in range(12):

    if jnp.linalg.norm(Voqq_mm[o]) > 1e-15:
        print(o, jnp.linalg.norm(Voqq_mm[o]), jnp.linalg.norm(Voqq_mm[o]-Voqq_mm[o].T))

    if jnp.linalg.norm(Voqq_pp[o]) > 1e-15:
        print(o, jnp.linalg.norm(Voqq_pp[o]), jnp.linalg.norm(Voqq_pp[o]-Voqq_pp[o].T))
        
    if jnp.linalg.norm(Voqq_mp[o]) > 1e-15:
        print(o, jnp.linalg.norm(Voqq_mp[o]), jnp.linalg.norm(Voqq_mp[o]-Voqq_mp[o].T))

    if jnp.linalg.norm(Voqq_pm[o]) > 1e-15:
        print(o, jnp.linalg.norm(Voqq_pm[o]), jnp.linalg.norm(Voqq_pm[o]-Voqq_pm[o].T))
        

    print(o, jnp.linalg.norm(Voqq_mm[o]-Voqq_mm[o].T) < 1e-15)
    print(o, jnp.linalg.norm(Voqq_pp[o]-Voqq_pp[o].T) < 1e-15)
    print(o, jnp.linalg.norm(Voqq_pm[o]-Voqq_mp[o].T) < 1e-15)
        
    '''
    fig, ax = plt.subplots(2, 2, figsize=(8,8))
    
    ax[0,0].imshow(Voqq_mm[o])
    ax[1,1].imshow(Voqq_pp[o])
    ax[0,1].imshow(Voqq_mp[o])
    ax[1,0].imshow(Voqq_pm[o])
    
    plt.show()
    '''
    
# full potential from affine decomposition
Vqq_mm_from_affine = jnp.einsum('o,oij->ij', LECs, Voqq_mm)
Vqq_pp_from_affine = jnp.einsum('o,oij->ij', LECs, Voqq_pp)
Vqq_mp_from_affine = jnp.einsum('o,oij->ij', LECs, Voqq_mp)
Vqq_pm_from_affine = jnp.einsum('o,oij->ij', LECs, Voqq_pm)


    
# full potential obtained directly from chiralms
Vqq_mm_direct = jnp.zeros((num_pts, num_pts))
Vqq_pp_direct = jnp.zeros((num_pts, num_pts))
Vqq_mp_direct = jnp.zeros((num_pts, num_pts))
Vqq_pm_direct = jnp.zeros((num_pts, num_pts))

for i, qi in enumerate(q_arr):
    for j, qj in enumerate(q_arr):
    
        Vqq_mm_direct = Vqq_mm_direct.at[i,j].set( V_chiral(qi, qj, channel_mm) )
        Vqq_pp_direct = Vqq_pp_direct.at[i,j].set( V_chiral(qi, qj, channel_pp) )
        Vqq_mp_direct = Vqq_mp_direct.at[i,j].set( V_chiral(qi, qj, channel_mp) )
        Vqq_pm_direct = Vqq_pm_direct.at[i,j].set( V_chiral(qi, qj, channel_pm) )
        
print(jnp.max(jnp.abs(Vqq_mm_direct - Vqq_mm_from_affine)))
print(jnp.max(jnp.abs(Vqq_pp_direct - Vqq_pp_from_affine)))
print(jnp.max(jnp.abs(Vqq_mp_direct - Vqq_mp_from_affine)))
print(jnp.max(jnp.abs(Vqq_pm_direct - Vqq_pm_from_affine)))

fig, ax = plt.subplots(2, 2, figsize=(8,8))

ax[0,0].imshow(jnp.abs(Vqq_mm_direct - Vqq_mm_from_affine))
ax[1,1].imshow(jnp.abs(Vqq_pp_direct - Vqq_pp_from_affine))
ax[0,1].imshow(jnp.abs(Vqq_mp_direct - Vqq_mp_from_affine))
ax[1,0].imshow(jnp.abs(Vqq_pm_direct - Vqq_pm_from_affine))

plt.show()


