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

fontsize = 10
black = 'k'

mpl.rcdefaults()  # Set to defaults

mpl.rc('text', usetex=True)
mpl.rc('text.latex', preamble=r'\usepackage{amssymb}\usepackage{fdsymbol}')
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

mpl.rcParams['axes.labelsize'] = fontsize
mpl.rcParams['axes.edgecolor'] = black
mpl.rcParams['axes.labelcolor'] = black
mpl.rcParams['axes.titlesize'] = fontsize

mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['xtick.color'] = black
mpl.rcParams['ytick.color'] = black
mpl.rcParams['xtick.major.width'] = mpl.rcParams['axes.linewidth'] * 0.95
mpl.rcParams['ytick.major.width'] = mpl.rcParams['axes.linewidth'] * 0.95
mpl.rcParams['xtick.minor.size'] = 2.4
mpl.rcParams['ytick.minor.size'] = 2.4
mpl.rcParams['xtick.major.size'] = 3.9
mpl.rcParams['ytick.major.size'] = 3.9
mpl.rcParams["xtick.minor.visible"] =  True
mpl.rcParams["ytick.minor.visible"] =  True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.top'] = True

ppi = 72  # points per inch
mpl.rcParams['figure.titlesize'] = fontsize
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['figure.constrained_layout.use'] = True
mpl.rcParams['figure.constrained_layout.wspace'] = 0.0
mpl.rcParams['figure.constrained_layout.hspace'] = 0.0
mpl.rcParams['figure.constrained_layout.h_pad'] = 3. / ppi
mpl.rcParams['figure.constrained_layout.w_pad'] = 3. / ppi

mpl.rcParams['legend.title_fontsize'] = fontsize
mpl.rcParams['legend.fontsize'] = fontsize
mpl.rcParams['legend.edgecolor'] = 'inherit'
mpl.rcParams['legend.facecolor'] = (1, 1, 1, 0.6)
mpl.rcParams['legend.fancybox'] = True
mpl.rcParams['legend.borderaxespad'] = 0.8
mpl.rcParams['legend.framealpha'] = None
mpl.rcParams['patch.linewidth'] = 0.8
mpl.rcParams['hatch.linewidth'] = 0.5
mpl.rcParams["axes.axisbelow"] = False

import matplotlib.pyplot as plt
color_68 = 'darkgrey'
color_95 = 'lightgrey'
mpt_default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
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

# data
filename = "benchmark/np_SGT/PWA93_0.1_300.txt"
Elabs_all, sigmas_all = np.loadtxt(filename, unpack=True)
indices = [0,13,18,28,38,48,58,68,78,88,98,108]
#indices = [0,5,10,20,30,40,50,60,70,80,90,100]
Elabs = Elabs_all[indices]
sigmas = sigmas_all[indices]

n_mesh = 40
Jmax = 4
static_indices = [0, 10, 11]

mesh = TRNS(n_mesh=n_mesh)
channels = Channels(isospin_channel='np', Jmax=Jmax)
potential = Chiral()
#solver = Solver(mesh, channels, potential, Elabs)
#emulator = Emulator(solver)

filename = "saved_samples/emulator_samples_0.1_0.1_Jmax4_Nq40_corr.npz"
data = jnp.load(filename)
LECs_em = data["LECs_em"]
sigmas_em = data["sigmas_em"]
err_sigmas_em = data["err_sigmas_em"]
params_c_em = data["params_c_em"]
params_cc_em = data["params_cc_em"]
best_fit_LECs = data["best fit LECs"]
MAP_LECs_em = data["MAP LECs"]

filename = "saved_samples/solver_samples_0.1_0.1_Jmax4_Nq40_corr.npz"
data = jnp.load(filename)
LECs_ex = data["LECs_ex"]
sigmas_ex = data["sigmas_ex"]
err_sigmas_ex = data["err_sigmas_ex"]
params_c_ex = data["params_c_ex"]
params_cc_ex = data["params_cc_ex"]
best_fit_LECs = data["best fit LECs"]
MAP_LECs_ex = data["MAP LECs"]


labels = [r"$C_S$", r"$C_T$", r"$C_1$", r"$C_2$", r"$C_3$", r"$C_4$", r"$C_5$", r"$C_6$", r"$C_7$"]

# -------------------------------
# Helper: overlay custom titles
# -------------------------------
def add_corner_titles(fig, samples, labels, color="k", yoffset=0.06):
    med = np.median(samples, axis=0)
    low = np.percentile(samples, 2.5, axis=0)
    up = np.percentile(samples, 97.5, axis=0)
    delta_low = med - low
    delta_up = up - med
    quant_texts = [
        fr"{med[i]:.3f}$^{{+{delta_up[i]:.3f}}}_{{-{delta_low[i]:.3f}}}$"
        for i in range(len(labels))
    ]
    axes = np.array(fig.axes).reshape(len(labels), len(labels))
    for i, ax in enumerate(np.diag(axes)):
        ax.text(
            0.5, 1.05 + yoffset, quant_texts[i],
            color=color, ha="center", va="bottom", transform=ax.transAxes,
            fontsize=fontsize
        )

# -------------------------------
# Corner plots
# -------------------------------
fig = plt.figure(figsize=(12, 12), constrained_layout=False)

corner.corner(
    np.array(LECs_em[:, 1:10]),
    range=[0.996] * (np.array(LECs_em[:, 1:10])).shape[1],
    color="k",
    smooth=True,
    fig=fig,
    labels=labels,
    labelpad=0.1,
    hist_kwargs={"density": True},
    contour_kwargs={"linestyles": "solid", "linewidths": 0.9},
)

corner.corner(
    np.array(LECs_ex[:, 1:10]),
    range=[0.996] * (np.array(LECs_ex[:, 1:10])).shape[1],
    color="C0",
    smooth=True,
    fig=fig,
    hist_kwargs={"density": True},
    contour_kwargs={"linestyles": "dashed", "linewidths": 0.9},
)



# --------------------------------
# Overlay best-fit and MAP points
# --------------------------------
corner.overplot_lines(fig, np.array(best_fit_LECs[1:10]), color="C3", ls="solid")
corner.overplot_points(fig, [np.array(best_fit_LECs[1:10])], marker="^", color="C3")

corner.overplot_lines(fig, np.array(MAP_LECs_ex[1:10]), color="C0", ls="dashed")
corner.overplot_points(fig, [np.array(MAP_LECs_ex[1:10])], marker="s", color="C0")

corner.overplot_lines(fig, np.array(MAP_LECs_em[1:10]), color="k", ls="dotted")
corner.overplot_points(fig, [np.array(MAP_LECs_em[1:10])], marker="o", color="k")




fig.subplots_adjust(wspace=0.15, hspace=0.15, left=0.1, right=0.95, top=0.95, bottom=0.1)

for ax in fig.get_axes():
    ax.label_outer()
    ax.tick_params(axis='both', pad=2)


    
# add emulator (black) stats titles on top
add_corner_titles(fig, np.array(LECs_em[:, 1:10]), labels, color="k", yoffset=0.0)

add_corner_titles(fig, np.array(LECs_ex[:, 1:10]), labels, color="C0", yoffset=0.2)

#plt.show()
plt.savefig("dnp_figures/corner.pdf", format="pdf")
# plt.close()

