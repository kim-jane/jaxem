import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
from jaxem import *
from collections import defaultdict
import corner


mesh = TRNS(n_mesh=40)
channels = Channels(isospin_channel='np', Jmax=4)
potential = Chiral()
solver = Solver(mesh, channels, potential)
emulator = Emulator(solver)


filename = "test/benchmark/np_SGT/PWA93_1_300.txt"
Elabs, sigmas = np.loadtxt(filename, unpack=True)
indices = [0,1,2,5,10,20,50,100,200,300]
Elabs = Elabs[indices]
sigmas_data = sigmas[indices]
    
#plt.plot(Elabs, sigmas_data, marker='o')
#plt.show()


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



sampler = Sampler(emulator, all_models, all_metas, Elabs, sigmas_data, static_indices=[0,10,11])

t = time.time()
LECs_samples, sigmas_samples, acc_rate = sampler.sample()
print("LECS_samples = ", LECs_samples.shape)
print("sigmas_samples = ", sigmas_samples.shape)
print("acceptance = ", acc_rate)

print("time to sample = ", time.time() - t)

filename = f"saved_samples/chiral_Nq{mesh.n_mesh}_Jmax{channels.Jmax}.npz"
jnp.savez(
    filename,
    LECs_samples=LECs_samples,
    sigmas_samples=sigmas_samples,
    Elabs=Elabs,
    sigmas_data=sigmas_data
)
