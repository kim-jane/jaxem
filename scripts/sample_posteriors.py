import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
from jaxem import *
from collections import defaultdict
import corner
print(jax.__version__)


# data
filename = "benchmark/np_SGT/PWA93_0.1_300.txt"
Elabs, sigmas = np.loadtxt(filename, unpack=True)
#indices = [0,1,2,5,10,20,50,100,200,300]
indices = [0,1,2,3,4,5,10,20,30,40,50,100]
Elabs = Elabs[indices]
sigmas = sigmas[indices]



#############

n_mesh = 40
Jmax = 2
static_indices = [0, 10, 11]

mesh = TRNS(n_mesh=n_mesh)
channels = Channels(isospin_channel='np', Jmax=Jmax)
potential = Chiral()
solver = Solver(mesh, channels, potential, Elabs)
emulator = Emulator(solver)


# train emulator
key = jax.random.PRNGKey(657)
key, subkey = jax.random.split(key)
LECs_candidates = sample_LECs(
    subkey,
    500,
    potential.LECs,
    scale_min=0.0,
    scale_max=2.0,
    static_indices=static_indices,
)


emulator.fit(LECs_candidates)

# prior predictive

prior_scale = 1.0 # 0.5
likelihood_scale = 0.1

sampler = Sampler(
    emulator, sigmas,
    prior_scale=prior_scale,
    likelihood_scale=likelihood_scale,
    static_indices=static_indices
)

n_chains = 20
n_equil = 10
n_skip = 10
n_samples_per_chain = 1000
init_noise = 0.1
step_scale = 0.1

start_time = time.time()


        
LECs_em, sigmas_em, params_c_em, params_cc_em, accept_rate_em = sampler.sample(
    n_chains=n_chains,
    n_equil=n_equil,
    n_skip=n_skip,
    n_samples_per_chain=n_samples_per_chain,
    init_noise=init_noise,
    step_scale=step_scale
)

delta_s, eta_s, sigma_c_s = params_c_em
delta_minus_s, delta_plus_s, epsilon_s, eta_minus_s, eta_plus_s, sigma_cc_s = params_cc_em

print("Emulator acceptance rate = ", accept_rate_em)
print("Time = ", time.time() - start_time)

fig = corner.corner(
    np.array(LECs_em[:,1:10]),
    color="k",
    smooth=True,
    show_titles=True,
    hist_kwargs={"density": True},
    contour_kwargs={"linestyles": "solid", "linewidths": 1.2}
)

start_time = time.time()
'''
LECs_ex, sigmas_ex, accept_rate_ex = sampler.sample(
    n_chains=n_chains,
    n_equil=n_equil,
    n_skip=n_skip,
    n_samples_per_chain=n_samples_per_chain,
    init_noise=init_noise,
    step_scale=step_scale,
    use_emulator=False,
)
'''
LECs_ex, sigmas_ex, accept_rate_ex = LECs_em, sigmas_em, accept_rate_em

print("Solver acceptance rate = ", accept_rate_ex)
print("Time = ", time.time() - start_time)

corner.corner(
    np.array(LECs_ex[:,1:10]),
    color="C0",
    smooth=True,
    fig=fig,
    hist_kwargs={"density": True},
    contour_kwargs={"linestyles": "dashed", "linewidths": 1.2}
)

corner.overplot_lines(fig, np.array(potential.LECs[1:10]), color="k", ls="--")
corner.overplot_points(fig, [np.array(potential.LECs[1:10])], marker="o", color="k")
corner.overplot_lines(fig, np.array(sampler.MAP_LECs[1:10]), color="r", ls="--")
corner.overplot_points(fig, [np.array(sampler.MAP_LECs[1:10])], marker="s", color="r")

plt.savefig(f"figures/test_corner_{prior_scale:.1f}_{likelihood_scale:.1f}.pdf", format="pdf")
plt.close()

def stats(arr):
    return jnp.mean(arr, axis=0), jnp.median(arr, axis=0), jnp.std(arr, axis=0)
    
    
avg_em, med_em, std_em = stats(sigmas_em)
avg_ex, med_ex, std_ex = stats(sigmas_ex)

plt.plot(Elabs, sigmas, color='k')

plt.plot(Elabs, med_em, color='C0')
plt.fill_between(Elabs, med_em-2*std_em, med_em+2*std_em, color='C0', alpha=0.2)

plt.plot(Elabs, med_ex, color='C1', linestyle='dashed')
plt.fill_between(Elabs, med_ex-2*std_ex, med_ex+2*std_ex, color='C1', alpha=0.2)

t = solver.onshell_t(potential.LECs)
sigma_tot = solver.scattering_params(t)[0]
plt.plot(Elabs, sigma_tot, color='k', linestyle='dashed')


#plt.xscale('log')
plt.yscale('log')
plt.savefig(f"figures/test_cross_sections_{prior_scale:.1f}_{likelihood_scale:.1f}.pdf", format="pdf")
plt.close()


avg_delta, med_delta, std_delta = stats(delta_s)
avg_delta_m, med_delta_m, std_delta_m = stats(delta_minus_s)
avg_delta_p, med_delta_p, std_delta_p = stats(delta_plus_s)
avg_epsilon, med_epsilon, std_epsilon = stats(epsilon_s)

for c, label in enumerate(channels.single_labels):
    plt.plot(Elabs, avg_delta[c])
    plt.fill_between(Elabs, avg_delta[c]-std_delta[c], avg_delta[c]+std_delta[c], alpha=0.5)
    plt.savefig(f"figures/test_{label}.pdf", format="pdf")
    plt.close()
    
for cc, label in enumerate(channels.coupled_labels):

    fig, axes = plt.subplots(1, 3, figsize=(12,4))
    axes[0].plot(Elabs, avg_delta_m[cc])
    axes[0].fill_between(Elabs, avg_delta_m[cc]-std_delta_m[cc], avg_delta_m[cc]+std_delta_m[cc], alpha=0.5)

    axes[1].plot(Elabs, avg_delta_p[cc])
    axes[1].fill_between(Elabs, avg_delta_p[cc]-std_delta_p[cc], avg_delta_p[cc]+std_delta_p[cc], alpha=0.5)
    
    axes[2].plot(Elabs, avg_epsilon[cc])
    axes[2].fill_between(Elabs, avg_epsilon[cc]-std_epsilon[cc], avg_epsilon[cc]+std_epsilon[cc], alpha=0.5)
    
    plt.savefig(f"figures/test_{label}.pdf", format="pdf")
    plt.close()
