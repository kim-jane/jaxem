import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jaxem import *
import matplotlib.pyplot as plt
import time
from collections import defaultdict


Elabs = jnp.linspace(1.0, 300., 10)


mesh = TRNS(n_mesh=40)
channels = Channels(isospin_channel='np', Jmax=1)
potential = Chiral()
solver = Solver(mesh, channels, potential)
emulator = Emulator(solver)

key = jax.random.PRNGKey(317)
key, subkey = jax.random.split(key)
LECs_candidates = sample_LECs(
    subkey,
    1000,
    potential.LECs,
    scale_min=0.0,
    scale_max=2.0,
    static_indices=[0, 10, 11]
)



for rom, c, ls in zip(['g', 'lspg'], ['C0', 'C1'], ['--', '..']):
    for mode in ['pod', 'greedy']:
        for Elab in Elabs:
            for channel in channels.all:
            
                # train the emulators
                model, errors, times = emulator.fit(
                    LECs_candidates,
                    channel=channel,
                    Elab=Elab,
                    tol=1e-3,
                    rom=rom,
                    mode=model,
                    n_init=2,
                    n_max=50,
                )
                
            
            sigmas = emulator.emulate_total_cross_sections(Elab)






print(f"channel = {channel}")
print(f"Elab = {Elab} MeV")
print(f"ROM = {rom}")
print(f"n_trials = {n_trials}")
print(f"seed = {seed}\n")




def print_dict(dict):
    for key, value in dict.items():
        print(f"{key}:")
        print(f"    {value}")
    print("")


for n_mesh in [40]:
    
    print(f"\nn_mesh = {n_mesh}")



    #linear_system = solver.setup_single_channel(channel, Elab)
    #linear_system = solver.setup_coupled_channel(channel, Elab)

    trials = {}

    for t in range(n_trials+2):
        
        key, subkey = jax.random.split(key)
        
        LECs_candidates = sample_LECs(
            subkey,
            1000,
            potential.LECs,
            scale_min=0.0,
            scale_max=2.0,
            static_indices=[0, 10, 11]
        )
        
        model, errors, times = emulator.fit(
            LECs_candidates,
            channel=channel,
            Elab=Elab,
            tol=1e-3,
            rom=rom,
            mode="greedy",
            n_init=2,
            n_max=21,
        )
        
        #print(f"\n{t}")
        #print_dict(model)
        #print_dict(errors)
        #print_dict(times)

        plt.semilogy(errors['min'])
        plt.semilogy(errors['p5'])
        plt.semilogy(errors['p25'])
        plt.semilogy(errors['med'])
        plt.semilogy(errors['p75'])
        plt.semilogy(errors['p95'])
        plt.semilogy(errors['max'])
        

        model, errors, times = emulator.fit(
            LECs_candidates,
            channel=channel,
            Elab=Elab,
            tol=1e-3,
            rom=rom,
            mode="pod",
        )
        
        print(f"\n{t}")
        print_dict(model)
        print_dict(errors)
        print_dict(times)

        plt.axhline(errors['min'])
        plt.axhline(errors['p5'])
        plt.axhline(errors['p25'])
        plt.axhline(errors['med'])
        plt.axhline(errors['p75'])
        plt.axhline(errors['p95'])
        plt.axhline(errors['max'])
        plt.show()

    '''
    trials[t] = times
        
    final_stats = {}

    t_range = range(2, n_trials+2)

    for key in trials[0].keys():
        final_stats["avg "+key] = np.mean(np.array([trials[t][key] for t in t_range]))
        final_stats["std "+key] = np.std(np.array([trials[t][key] for t in t_range]), ddof=1)

    for name, value in final_stats.items():
        print(f"{name}: {value}")
        
    '''



