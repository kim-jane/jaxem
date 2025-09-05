import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
from jaxem import *
from collections import defaultdict



mesh = TRNS(n_mesh=40)
channels = Channels(isospin_channel='np', Jmax=1)
potential = Chiral()
solver = Solver(mesh, channels, potential)
emulator = Emulator(solver)


filename = "test/benchmark/np_SGT/PWA93_1_300.txt"
Elabs, sigmas = np.loadtxt(filename, unpack=True)
indices = [0,1,2,5,10,50,100,200]
Elabs = Elabs[indices]
sigmas_data = sigmas[indices]
    
plt.plot(Elabs, sigmas_data, color='k')
    

key = jax.random.PRNGKey(317)
key, subkey = jax.random.split(key)
LECs_candidates = sample_LECs(
    subkey,
    500,
    potential.LECs,
    scale_min=0.0,
    scale_max=2.0,
    static_indices=[0, 10, 11]
)

# train all the emulators
all_models = []
all_metas = []

for Elab in Elabs:

    models_for_Elab = []
    metas_for_Elab = []
    
    for channel in channels.all.keys():
    
        model, errors, times, meta = emulator.fit(
            LECs_candidates,
            channel=channel,
            Elab=Elab,
            tol=1e-5,
            rom='g',
            mode='pod',
            n_max=40
        )
        
        models_for_Elab.append( model )
        metas_for_Elab.append( meta )
        
        
    all_models.append( models_for_Elab )
    all_metas.append( metas_for_Elab )


sampler = Sampler(emulator, all_models, all_metas, Elabs, sigmas_data)

t = time.time()
LECs_samples = sampler.sample(use_emulators=False)
print("Time to sample solved = ", time.time()-t)
print("LECS_samples = ", LECs_samples.shape)

t = time.time()
LECs_samples = sampler.sample(use_emulators=True)
print("Time to sample emulated = ", time.time()-t)
print("LECS_samples = ", LECs_samples.shape)




n_energies = Elabs.shape[0]
n_samples = LECs_samples.shape[0]

sigmas_em = jnp.zeros((n_energies, n_samples))

for i in range(n_energies):
    for j in range(n_samples):

        sigma = emulator.total_cross_section(
            all_models[i],
            all_metas[i],
            LECs=LECs_samples[j]
        )
    
        sigmas_em = sigmas_em.at[i,j].set(sigma.squeeze())
    

for j in range(n_samples):
    plt.scatter(Elabs, 10. * sigmas_em[:,j], color='b', marker='o', s=0.5)

plt.show()

