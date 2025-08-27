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

Elab = 10.0
channel = '3S1-3D1'

key = jax.random.PRNGKey(317)

for i in range(1):

    key, subkey = jax.random.split(key)
    LECs_candidates = sample_LECs(
        subkey,
        500,
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
        rom='lspg',
        mode='greedy',
        n_init=10
    )

    print(model["trial basis"].shape)
    t_em = emulator.t(
        model,
        LECs=potential.LECs
    )
    print("t_em = ", t_em.shape)
    #plt.plot(t_em.real, linestyle='dashed')


    t_ex = solver.t(
        channel=channel,
        Elab=Elab,
        LECs=potential.LECs
    )
    print("t_ex = ", t_ex.shape)

    #plt.plot(t_ex.real)
    #plt.plot(t_em.real, linestyle='dashed')
    #plt.show()



    plt.plot(t_ex[0,0].real)
    plt.plot(t_ex[1,1].real)
    plt.plot(t_ex[0,1].real)
    plt.plot(t_ex[1,0].real)

    plt.plot(t_em[0,0].real, linestyle='dashed')
    plt.plot(t_em[1,1].real, linestyle='dashed')
    plt.plot(t_em[0,1].real, linestyle='dashed')
    plt.plot(t_em[1,0].real, linestyle='dashed')

    plt.show()

