import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
from jaxem import *
from collections import defaultdict
import corner




n_mesh = 60
Jmax = 2
static_indices = [0, 10, 11]
n_candidates = 500
n_trials = 100
n_warmup = 3

Elabs = [0.1, 1., 2., 5., 10., 20., 50., 100., 200., 300.]

mesh = TRNS(n_mesh=n_mesh)
channels = Channels(isospin_channel='np', Jmax=Jmax)
potential = Chiral()
solver = Solver(mesh, channels, potential, Elabs)
emulator = Emulator(solver)


# train emulator
key = jax.random.PRNGKey(317)
key, subkey = jax.random.split(key)
LECs_candidates = sample_LECs(
    subkey,
    n_candidates,
    potential.LECs,
    scale_min=0.0,
    scale_max=2.0,
    static_indices=static_indices
)

emulator.fit(LECs_candidates, n_max=50)



# containers for times
times_em = jnp.zeros((n_trials,))
times_ex = jnp.zeros((n_trials,))


for _ in range(n_warmup):

    t = solver.t(potential.LECs)
    sigma_tot = solver.total_cross_sections(t)

for i in range(n_trials):

    start_time = time.time()
    
    t = solver.t(potential.LECs)
    sigma_tot = solver.total_cross_sections(t)

    end_time = time.time()
    
    times_ex = times_ex.at[i].set( end_time - start_time )
    
    if i % 10 == 0:
        print(i)
        
        

for _ in range(n_warmup):

    t = emulator.t(potential.LECs)
    sigma_tot = solver.total_cross_sections(t)

for i in range(n_trials):

    start_time = time.time()
    
    t = emulator.t(potential.LECs)
    sigma_tot = solver.total_cross_sections(t)

    end_time = time.time()
    
    times_em = times_em.at[i].set( end_time - start_time )
    
    if i % 10 == 0:
        print(i)






avg_time_em = jnp.mean(times_em)
avg_time_ex = jnp.mean(times_ex)
std_time_em = jnp.std(times_em)
std_time_ex = jnp.std(times_ex)

avg_speedup = avg_time_ex / avg_time_em
std_speedup = avg_speedup * jnp.sqrt( (std_time_ex / avg_time_ex)**2 + (std_time_em / avg_time_em)**2 )

print(f"Emulation time: {avg_time_em} +- {std_time_em}")
print(f"High-fidelity time: {avg_time_ex} +- {std_time_ex}")
print(f"Speedup: {avg_speedup} +- {std_speedup}")


"""

n_mesh = 20
Jmax = 4
n_trials = 100
Emulation time: 0.004194869995117188 +- 0.0008048244831640177
High-fidelity time: 0.021473500728607178 +- 0.004921853775017479
Speedup: 5.118990756233744 +- 1.5301015303527705

n_mesh = 40
Jmax = 4
n_trials = 100
Emulation time: 0.004355447292327881 +- 0.0008387569680783573
High-fidelity time: 0.16539162874221802 +- 0.03585498878770013
Speedup: 37.973511706491166 +- 11.011201380553965

n_mesh = 60
Jmax = 4
n_trials = 100
Emulation time: 0.004546430110931397 +- 0.0009012021378486508
High-fidelity time: 0.6637035059928894 +- 0.14232732183416366
Speedup: 145.98343970956168 +- 42.63071830652379

n_mesh = 80
Jmax = 4
n_trials = 100
Emulation time: 0.004543344974517823 +- 0.0008431148442315965
High-fidelity time: 1.2739479207992555 +- 0.26476068677348313
Speedup: 280.3986771738497 +- 78.12451330488844

n_mesh = 120
Jmax = 4
n_trials = 100
Emulation time: 0.00491302490234375 +- 0.0009974865710791036
High-fidelity time: 2.3284021401405335 +- 0.4868624566694136
Speedup: 473.9243513766383 +- 138.12470785246597

n_mesh = 160
Jmax = 4
n_trials = 100
Emulation time: 0.005109994411468506 +- 0.0011177749690276758
High-fidelity time: 4.2830679631233215 +- 0.9560467587366628
Speedup: 838.1746863579166 +- 261.95281847625984



n_mesh = 20
Jmax = 2
n_trials = 100
Emulation time: 0.002087588310241699 +- 0.0004128037126818148
High-fidelity time: 0.012013189792633057 +- 0.0027621907513908288
Speedup: 5.754578014111499 +- 1.7451614789652932

n_mesh = 40
Jmax = 2
n_trials = 100
Emulation time: 0.002081012725830078 +- 0.00040621548586175044
High-fidelity time: 0.07718876838684083 +- 0.015745307552941956
Speedup: 37.091925209660424 +- 10.472347180745555

n_mesh = 60
Jmax = 2
n_trials = 100
Emulation time: 0.002273550033569336 +- 0.0004637928128810689
High-fidelity time: 0.28624667882919314 +- 0.06095518430600315
Speedup: 125.90295995369108 +- 37.12752161653226

n_mesh = 80
Jmax = 2
n_trials = 100
Emulation time: 0.0022428393363952636 +- 0.00043961290751037775
High-fidelity time: 0.6727320265769958 +- 0.14008114160690324
Speedup: 299.9465970033432 +- 85.7750013711419

n_mesh = 120
Jmax = 2
n_trials = 100
Emulation time: 0.0023093414306640625 +- 0.00043901166290615987
High-fidelity time: 1.1912515044212342 +- 0.24430608534946943
Speedup: 515.8403554379067 +- 144.24932621497834

n_mesh = 160
Jmax = 2
n_trials = 100
Emulation time: 0.002598392963409424 +- 0.000303560975394371
High-fidelity time: 2.15634672164917 +- 0.4453215881366643
Speedup: 829.8770632521138 +- 196.9058146126312

"""
