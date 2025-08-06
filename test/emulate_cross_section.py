import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jaxem import *
import matplotlib.pyplot as plt
import time
from collections import defaultdict


n_mesh = 40
n_val = 1000
tol = 1e-5
Elab = 10.0
rom = 'g'
mode = 'greedy'
Jmax = 1


print(f"Elab = {Elab} MeV")
print(f"n_mesh = {n_mesh}")
print(f"n_val = {n_val}")
print(f"ROM = {rom}")
print(f"mode = {mode}")
print(f"Jmax = {Jmax}")


mesh = TRNS(n_mesh=n_mesh)
channels = Channels(isospin_channel='np', Jmax=1)
potential = Chiral()
solver = Solver(mesh, channels, potential)
emulator = Emulator(solver)


key = jax.random.PRNGKey(317)
key, subkey = jax.random.split(key)
LECs_candidates = sample_LECs(
    subkey,
    n_val,
    potential.LECs,
    scale_min=0.5,
    scale_max=1.5,
    static_indices=[0, 10, 11]
)


for channel in channels.coupled.keys():

    print(f"\nchannel = {channel}")
    
    '''
    model, errors, times = emulator.fit(
        LECs_candidates,
        channel=channel,
        Elab=Elab,
        tol=tol,
        rom=rom,
        mode=mode,
        n_max=25,
    )
    '''
    
    t = solver.t(channel, Elab, potential.LECs)
    
    obs = solver.observables(channel, Elab, t)
    
