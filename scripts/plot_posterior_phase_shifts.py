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


import matplotlib as mpl

fontsize = 9
black = 'k'

mpl.rcdefaults()  # Set to defaults

mpl.rc('text', usetex=True)
mpl.rc('text.latex', preamble=r'\usepackage{amssymb}\usepackage{fdsymbol}')
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

mpl.rcParams['axes.labelsize'] = fontsize
mpl.rcParams['axes.edgecolor'] = black
# mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.labelcolor'] = black
mpl.rcParams['axes.titlesize'] = fontsize

mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['xtick.color'] = black
mpl.rcParams['ytick.color'] = black
# Make the ticks thin enough to not be visible at the limits of the plot (over the axes border)
mpl.rcParams['xtick.major.width'] = mpl.rcParams['axes.linewidth'] * 0.95
mpl.rcParams['ytick.major.width'] = mpl.rcParams['axes.linewidth'] * 0.95
# The minor ticks are little too small, make them both bigger.
mpl.rcParams['xtick.minor.size'] = 2.4  # Default 2.0
mpl.rcParams['ytick.minor.size'] = 2.4
mpl.rcParams['xtick.major.size'] = 3.9  # Default 3.5
mpl.rcParams['ytick.major.size'] = 3.9
mpl.rcParams["xtick.minor.visible"] =  True
mpl.rcParams["ytick.minor.visible"] =  True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.top'] = True

ppi = 72  # points per inch
# dpi = 150
mpl.rcParams['figure.titlesize'] = fontsize
mpl.rcParams['figure.dpi'] = 150  # To show up reasonably in notebooks
mpl.rcParams['figure.constrained_layout.use'] = True
# 0.02 and 3 points are the defaults:
# can be changed on a plot-by-plot basis using fig.set_constrained_layout_pads()
mpl.rcParams['figure.constrained_layout.wspace'] = 0.0
mpl.rcParams['figure.constrained_layout.hspace'] = 0.0
mpl.rcParams['figure.constrained_layout.h_pad'] = 3. / ppi  # 3 points
mpl.rcParams['figure.constrained_layout.w_pad'] = 3. / ppi

mpl.rcParams['legend.title_fontsize'] = fontsize
mpl.rcParams['legend.fontsize'] = fontsize
mpl.rcParams['legend.edgecolor'] = 'inherit'  # inherits from axes.edgecolor, to match
mpl.rcParams['legend.facecolor'] = (1, 1, 1, 0.6)  # Set facecolor with its own alpha, so edgecolor is unaffected
mpl.rcParams['legend.fancybox'] = True
mpl.rcParams['legend.borderaxespad'] = 0.8
mpl.rcParams['legend.framealpha'] = None  # Do not set overall alpha (affects edgecolor). Handled by facecolor above
mpl.rcParams['patch.linewidth'] = 0.8  # This is for legend edgewidth, since it does not have its own option

mpl.rcParams['hatch.linewidth'] = 0.5
mpl.rcParams["axes.axisbelow"] = False

import matplotlib.pyplot as plt
color_68 = 'darkgrey'   # color for 1 sigma bands
color_95 = 'lightgrey'  # color for 2 sigma bands
mpt_default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']  # 8 colors
colorset = ['b', 'r', 'darkcyan', 'darkslateblue', 'orange', 'darkgrey', 'lime', 'aqua', 'g', 'magenta', 'k']  +  colors
markers = {'0': "*", '2': "o", '3': "s", '4': "D", '5': "p" }

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
[purple, blue, grey, red, darkblue, green] = flatui
orange = '#f39c12'
black = 'k'
yellow = 'yellow'

color_list = ['Oranges', 'Greens', 'Blues', 'Reds', 'Purples', 'Greys', 'plasma']
cmaps = [plt.get_cmap(name) for name in color_list]
colors_alt = [cmap(0.55 - 0.1 * (i == 0)) for i, cmap in enumerate(cmaps)]
colors_alt2 = plt.get_cmap('tab20').colors


'''
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["STIX"],       # or "Times New Roman"
    "mathtext.fontset": "stix",   # ensures math matches the text
    "font.size": 12
})
'''


# data
filename = "benchmark/np_SGT/PWA93_0.1_300.txt"
Elabs_all, sigmas_all = np.loadtxt(filename, unpack=True)
indices = [0,13,18,28,38,48,58,68,78,88,98,108]
#indices = [0,5,10,20,30,40,50,60,70,80,90,100]
Elabs = Elabs_all[indices]
sigmas = sigmas_all[indices]

n_mesh = 40
Jmax = 2
static_indices = [0, 10, 11]

mesh = TRNS(n_mesh=n_mesh)
channels = Channels(isospin_channel='np', Jmax=Jmax)
potential = Chiral()
solver = Solver(mesh, channels, potential, Elabs)
emulator = Emulator(solver)


filename = "saved_samples/test_emulator_samples_0.2_0.1.npz"
data = jnp.load(filename)
LECs_em = data["LECs_em"]
sigmas_em = data["sigmas_em"]
err_sigmas_em = data["err_sigmas_em"]
params_c_em = data["params_c_em"]
params_cc_em = data["params_cc_em"]
best_fit_LECs = data["best fit LECs"]
MAP_LECs_em = data["MAP LECs"]

filename = "saved_samples/test_solver_samples_0.2_0.1.npz"
data = jnp.load(filename)
LECs_ex = data["LECs_ex"]
sigmas_ex = data["sigmas_ex"]
err_sigmas_ex = data["err_sigmas_ex"]
params_c_ex = data["params_c_ex"]
params_cc_ex = data["params_cc_ex"]
best_fit_LECs = data["best fit LECs"]
MAP_LECs_ex = data["MAP LECs"]


def stats(arr):
    median = jnp.median(arr, axis=0)
    lower, upper = jnp.percentile(arr, jnp.array([2.5, 97.5]), axis=0)
    return median, median-lower, upper-median
    
    
##################################
# CROSS SECTIONS
##################################

onshell_t_and_err = solver.onshell_t_and_err(potential.LECs)
_, _, params_c_best, params_cc_best = solver.scattering_params(onshell_t_and_err)

delta_best, eta_best, sigma_c_best = params_c_best
delta_minus_best, delta_plus_best, epsilon_best, eta_minus_best, eta_plus_best, sigma_cc_best = params_cc_best

delta_em, eta_em, sigma_c_em = params_c_em
delta_minus_em, delta_plus_em, epsilon_em, eta_minus_em, eta_plus_em, sigma_cc_em = params_cc_em

delta_ex, eta_ex, sigma_c_ex = params_c_ex
delta_minus_ex, delta_plus_ex, epsilon_ex, eta_minus_ex, eta_plus_ex, sigma_cc_ex = params_cc_ex


med_delta_em, low_delta_em, up_delta_em = stats(delta_em)
med_delta_m_em, low_delta_m_em, up_delta_m_em = stats(delta_minus_em)
med_delta_p_em, low_delta_p_em, up_delta_p_em = stats(delta_plus_em)
med_epsilon_em, low_epsilon_em, up_epsilon_em = stats(epsilon_em)

med_delta_ex, low_delta_ex, up_delta_ex = stats(delta_ex)
med_delta_m_ex, low_delta_m_ex, up_delta_m_ex = stats(delta_minus_ex)
med_delta_p_ex, low_delta_p_ex, up_delta_p_ex = stats(delta_plus_ex)
med_epsilon_ex, low_epsilon_ex, up_epsilon_ex = stats(epsilon_ex)

kwargs = {
    'markersize': 5,
    'capsize': 4,
    'elinewidth': 1,
    'linewidth': 0
}

def spectro_label(s):
    # assumes format like "1S0"
    return fr"$^{{{s[0]}}}{s[1]}_{{{s[2:]}}}$"
    
    
for c, label in enumerate(channels.single_labels):

    
    filename = f"benchmark/np_phase_shifts/PWA93_0.1_300_{label}.txt"
    Elabs_all, deltas_all = np.loadtxt(filename, unpack=True)
    Elabs = Elabs_all[indices]
    deltas = deltas_all[indices]
    
    print(len(deltas))
    print(len(Elabs))
    print(len(delta_best))
    
    print(label)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(4, 4), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )
    
    f = 180/np.pi

    ax1.plot(Elabs_all, deltas_all, color='k', label='PWA93')
    ax1.errorbar(Elabs, f*med_delta_ex[c], yerr=(f*low_delta_ex[c], f*up_delta_ex[c]), color='C0', marker='s', label=r'High-Fidelity (95\% CI)', **kwargs)
    ax1.errorbar(Elabs, f*med_delta_em[c], yerr=(f*low_delta_em[c], f*up_delta_em[c]), color='C1', marker='o', label=r'Low-Fidelity (95\% CI)', **kwargs)
    ax1.errorbar(Elabs, f*delta_best[c], color='C3', marker='^', label='Best Fit', **kwargs)

    ax2.axhline(0, color='k')
    ax2.errorbar(Elabs, (f*med_delta_ex[c] - deltas), yerr=(np.abs(f*low_delta_ex[c]), np.abs(f*up_delta_ex[c])), color='C0', marker='s', label='High-Fidelity', **kwargs)
    ax2.errorbar(Elabs, (f*med_delta_em[c] - deltas), yerr=(np.abs(f*low_delta_em[c]), np.abs(f*up_delta_em[c])), color='C1', marker='o', label='Low-Fidelity', **kwargs)
    ax2.errorbar(Elabs, (f*delta_best[c] - deltas), color='C3', marker='^', label='Best Fit', **kwargs)

    formatted_labels =0

    plt.xlim(-1, 101)

    plt.xlabel(r"$E_{lab}$ [MeV]")
    ax1.set_ylabel(r"$\delta$ [deg]")
    ax2.set_ylabel("Deviation")
    ax1.legend()
    #plt.show()
    plt.savefig(f"figures/test_phase_shifts_{label}.pdf", format="pdf")

    #plt.show()
    #plt.close()
    
'''
for cc, label in enumerate(channels.coupled_labels):

    fig, axes = plt.subplots(1, 3, figsize=(12,4))
    
    axes[0].plot(Elabs, med_delta_m_em[cc], color='C0')
    axes[0].plot(Elabs, med_delta_m_ex[cc], color='C1', linestyle='dashed')
    axes[0].errorbar(Elabs, med_delta_m_em[cc], yerr=(low_delta_m_em[cc], up_delta_m_em[cc]), color='C0', marker='s', label='Low-Fidelity', linewidth=0, elinewidth=2, capsize=10)
    axes[0].errorbar(Elabs, med_delta_m_ex[cc], yerr=(low_delta_m_ex[cc], up_delta_m_ex[cc]), color='C1', marker='^', label='High-Fidelity', linewidth=0, elinewidth=2, capsize=10)

    axes[1].plot(Elabs, med_delta_p_em[cc], color='C0')
    axes[1].plot(Elabs, med_delta_p_ex[cc], color='C1', linestyle='dashed')
    axes[1].errorbar(Elabs, med_delta_p_em[cc], yerr=(low_delta_p_em[cc], up_delta_p_em[cc]), color='C0', marker='s', label='Low-Fidelity', linewidth=0, elinewidth=2, capsize=10)
    axes[1].errorbar(Elabs, med_delta_p_ex[cc], yerr=(low_delta_p_ex[cc], up_delta_p_ex[cc]), color='C1', marker='^', label='High-Fidelity', linewidth=0, elinewidth=2, capsize=10)
    
    axes[2].plot(Elabs, med_epsilon_em[cc], color='C0')
    axes[2].plot(Elabs, med_epsilon_ex[cc], color='C1', linestyle='dashed')
    axes[2].errorbar(Elabs, med_epsilon_em[cc], yerr=(low_epsilon_em[cc], up_epsilon_em[cc]), color='C0', marker='s', label='Low-Fidelity', linewidth=0, elinewidth=2, capsize=10)
    axes[2].errorbar(Elabs, med_epsilon_ex[cc], yerr=(low_epsilon_ex[cc], up_epsilon_ex[cc]), color='C1', marker='^', label='High-Fidelity', linewidth=0, elinewidth=2, capsize=10)
    
    #plt.savefig(f"figures/test_{label}.pdf", format="pdf")
    plt.show()
    plt.close()



fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(4, 4), sharex=True,
    gridspec_kw={"height_ratios": [2, 1]}
)

med_em, low_em, up_em = stats(sigmas_em)
med_ex, low_ex, up_ex = stats(sigmas_ex)





ax1.plot(Elabs_all, sigmas_all, color='k', label='PWA93')
ax1.errorbar(Elabs, med_ex, yerr=(low_ex, up_ex), color='C0', marker='s', label=r'High-Fidelity (95\% CI)', **kwargs)
ax1.errorbar(Elabs, med_em, yerr=(low_em, up_em), color='C1', marker='o', label=r'Low-Fidelity (95\% CI)', **kwargs)
ax1.errorbar(Elabs, sigmas_best, color='C3', marker='^', label='Best Fit', **kwargs)

ax1.set_ylim(50, 20000)
ax1.set_yscale('log')


ax2.axhline(0, color='k')
ax2.errorbar(Elabs, (med_ex-sigmas)/sigmas, yerr=(low_ex/sigmas, up_ex/sigmas), color='C0', marker='s', label='High-Fidelity', **kwargs)
ax2.errorbar(Elabs, (med_em-sigmas)/sigmas, yerr=(low_em/sigmas, up_em/sigmas), color='C1', marker='o', label='Low-Fidelity', **kwargs)
ax2.errorbar(Elabs, (sigmas_best-sigmas)/sigmas, color='C3', marker='^', label='Best Fit', **kwargs)
ax2.set_ylim(-0.06, 0.06)


plt.xlim(-1, 101)

plt.xlabel(r"$E_{lab}$ [MeV]")
ax1.set_ylabel(r"$\sigma_{tot}$ [mb]")
ax2.set_ylabel("Relative Deviation")
ax1.legend()
#plt.show()
plt.savefig(f"figures/test_cross_sections.pdf", format="pdf")
plt.close()

print(Elabs)



'''
