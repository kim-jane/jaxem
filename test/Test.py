import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jaxem import *
import matplotlib.pyplot as plt
import time
from collections import defaultdict

n_trials = 100
channel = '1S0'
Elab = 10.0

print(f"n_trials = {n_trials}")
print(f"channel = {channel}")
print(f"Elab = {Elab} MeV")

channels = Channels(isospin_channel='np', Jmax=0)
potential = Chiral()


for n_mesh in [40]:
    
    print(f"\nn_mesh = {n_mesh}")

    mesh = TRNS(n_mesh=n_mesh, n_mesh1=int(0.625 * n_mesh))
    solver = Solver(mesh, channels, potential)
    emulator = Emulator(solver)

    linear_system = solver.setup_single_channel(channel, Elab)

    key = jax.random.PRNGKey(317)

    trials = {}

    for t in range(n_trials+2):
        
        key, subkey = jax.random.split(key)
        
        LECs_candidates = sample_LECs(
            subkey,
            1000,
            potential.LECs,
            scale_min=0.5,
            scale_max=1.5,
            static_indices=[]
        )

        errors, basis, LECs_snapshots, times = emulator.fit(
            LECs_candidates,
            channel=channel,
            Elab=Elab,
            n_init=2,
            n_max=25,
            err_tol=1e-8,
            rom="lspg",
            mode="pod",
            linear_system=linear_system
        )
        

        '''
        for name, value in times.items():
            print(f"{name}: {value}")
        print()
        '''

        trials[t] = times
        
    final_stats = {}

    t_range = range(2, n_trials+2)

    for key in trials[0].keys():
        final_stats["avg "+key] = np.mean(np.array([trials[t][key] for t in t_range]))
        final_stats["std "+key] = np.std(np.array([trials[t][key] for t in t_range]), ddof=1)

    for name, value in final_stats.items():
        print(f"{name}: {value}")
        
