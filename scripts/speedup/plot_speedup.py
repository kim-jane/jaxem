import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import plot_config

cm_to_in = 1. / 2.54
col_width = 8.55 * cm_to_in
text_width = 17.1 * cm_to_in
text_height = 22.9 * cm_to_in



solver_output_file = "scripts/speedup/results/output_solver.txt"

n_mesh_solver = []
times_solver = []
errs_solver = []

with open(solver_output_file, "r") as f:
    for line in f:
        if "!*!" in line:
        
            split_line = line[3:].split(",")
            
            n_mesh = int(split_line[0])
            time = float(split_line[1])
            err = float(split_line[2])
            
            n_mesh_solver.append( n_mesh )
            times_solver.append( time )
            errs_solver.append( err )
            
n_mesh_solver = np.array(n_mesh_solver)
times_solver = np.array(times_solver)
errs_solver = 10 * np.array(errs_solver)

print("n_mesh_solver = ", n_mesh_solver)
print("times_solver = ", times_solver)
print("errs_solver = ", errs_solver)


current_tol = None
current_rom = None
current_single = True
current_n_mesh = None

n_mesh_g_4 = []
n_mesh_g_5 = []
n_mesh_g_6 = []
n_mesh_g_7 = []

times_g_4 = []
times_g_5 = []
times_g_6 = []
times_g_7 = []

errs_g_4 = []
errs_g_5 = []
errs_g_6 = []
errs_g_7 = []

n_basis_g_4 = defaultdict(dict)
n_basis_g_5 = defaultdict(dict)
n_basis_g_6 = defaultdict(dict)
n_basis_g_7 = defaultdict(dict)

n_mesh_lspg_4 = []
n_mesh_lspg_5 = []
n_mesh_lspg_6 = []
n_mesh_lspg_7 = []

times_lspg_4 = []
times_lspg_5 = []
times_lspg_6 = []
times_lspg_7 = []

errs_lspg_4 = []
errs_lspg_5 = []
errs_lspg_6 = []
errs_lspg_7 = []

n_basis_lspg_4 = defaultdict(dict)
n_basis_lspg_5 = defaultdict(dict)
n_basis_lspg_6 = defaultdict(dict)
n_basis_lspg_7 = defaultdict(dict)


emulator_output_file = "scripts/speedup/results/output_emulators.txt"
with open(emulator_output_file, "r") as f:
    for line in f:
    
        if "TOLERANCE" in line:
            current_tol = float(line.split()[-1])
            
            
        elif "ROM" in line:
            current_rom = line.split()[-1]
            
            
        elif "MESH POINTS" in line:
            current_n_mesh = int(line.split()[-1])
            print(current_n_mesh)
            
            
        elif "(" in line and ")" in line:
            split_line = line[:-2].split()
            if len(split_line) == 4:
            
                if current_n_mesh == 80:
                
                    k = int(split_line[0])
                    c = int(split_line[1])
                    if current_single == False:
                        c += 14

                    #print(current_single, c)
                    n_basis = int(split_line[3])
                    
                    if current_tol == 1e-4 and current_rom == "g":
                        n_basis_g_4[k][c] = n_basis
                        
                    elif current_tol == 1e-5 and current_rom == "g":
                        n_basis_g_5[k][c] = n_basis
                        
                    elif current_tol == 1e-6 and current_rom == "g":
                        n_basis_g_6[k][c] = n_basis
                        
                    elif current_tol == 1e-7 and current_rom == "g":
                        n_basis_g_7[k][c] = n_basis
                        
                    elif current_tol == 1e-4 and current_rom == "lspg":
                        n_basis_lspg_4[k][c] = n_basis
                        
                    elif current_tol == 1e-5 and current_rom == "lspg":
                        n_basis_lspg_5[k][c] = n_basis
                        
                    elif current_tol == 1e-6 and current_rom == "lspg":
                        n_basis_lspg_6[k][c] = n_basis
                        
                    elif current_tol == 1e-7 and current_rom == "lspg":
                        n_basis_lspg_7[k][c] = n_basis
                    
                    if c == 13 and current_single == True:
                        current_single = False
                        
                    elif c == 19 and current_single == False:
                        current_single = True
                    
                
            
        elif "!*!" in line:
        
            split_line = line[3:].split(",")
            
            n_mesh = int(split_line[0])
            time = float(split_line[1])
            err = float(split_line[2])
            
            if current_tol == 1e-4 and current_rom == "g":
                n_mesh_g_4.append( n_mesh )
                times_g_4.append( time )
                errs_g_4.append( err )
                
            elif current_tol == 1e-5 and current_rom == "g":
                n_mesh_g_5.append( n_mesh )
                times_g_5.append( time )
                errs_g_5.append( err )
                
            elif current_tol == 1e-6 and current_rom == "g":
                n_mesh_g_6.append( n_mesh )
                times_g_6.append( time )
                errs_g_6.append( err )
                
            elif current_tol == 1e-7 and current_rom == "g":
                n_mesh_g_7.append( n_mesh )
                times_g_7.append( time )
                errs_g_7.append( err )
                
            elif current_tol == 1e-4 and current_rom == "lspg":
                n_mesh_lspg_4.append( n_mesh )
                times_lspg_4.append( time )
                errs_lspg_4.append( err )
                
            elif current_tol == 1e-5 and current_rom == "lspg":
                n_mesh_lspg_5.append( n_mesh )
                times_lspg_5.append( time )
                errs_lspg_5.append( err )
                
            elif current_tol == 1e-6 and current_rom == "lspg":
                n_mesh_lspg_6.append( n_mesh )
                times_lspg_6.append( time )
                errs_lspg_6.append( err )
                
            elif current_tol == 1e-7 and current_rom == "lspg":
                n_mesh_lspg_7.append( n_mesh )
                times_lspg_7.append( time )
                errs_lspg_7.append( err )
                
n_mesh_g_4 = np.array(n_mesh_g_4)
n_mesh_g_5 = np.array(n_mesh_g_5)
n_mesh_g_6 = np.array(n_mesh_g_6)
n_mesh_g_7 = np.array(n_mesh_g_7)
n_mesh_lspg_4 = np.array(n_mesh_lspg_4)
n_mesh_lspg_5 = np.array(n_mesh_lspg_5)
n_mesh_lspg_6 = np.array(n_mesh_lspg_6)
n_mesh_lspg_7 = np.array(n_mesh_lspg_7)

times_g_4 = np.array(times_g_4)
times_g_5 = np.array(times_g_5)
times_g_6 = np.array(times_g_6)
times_g_7 = np.array(times_g_7)
times_lspg_4 = np.array(times_lspg_4)
times_lspg_5 = np.array(times_lspg_5)
times_lspg_6 = np.array(times_lspg_6)
times_lspg_7 = np.array(times_lspg_7)

errs_g_4 = 10 * np.array(errs_g_4)
errs_g_5 = 10 * np.array(errs_g_5)
errs_g_6 = 10 * np.array(errs_g_6)
errs_g_7 = 10 * np.array(errs_g_7)
errs_lspg_4 = 10 * np.array(errs_lspg_4)
errs_lspg_5 = 10 * np.array(errs_lspg_5)
errs_lspg_6 = 10 * np.array(errs_lspg_6)
errs_lspg_7 = 10 * np.array(errs_lspg_7)

# FIX THIS LATER
n_mesh_g_7 = n_mesh_lspg_7
times_g_7 = times_lspg_7
errs_g_7 = errs_lspg_7
n_basis_g_7 = n_basis_lspg_7

print("n_mesh_g_4 = ", n_mesh_g_4)
print("n_mesh_g_5 = ", n_mesh_g_5)
print("n_mesh_g_6 = ", n_mesh_g_6)
print("n_mesh_g_7 = ", n_mesh_g_7)

print("times_g_4 = ", times_g_4)
print("times_g_5 = ", times_g_5)
print("times_g_6 = ", times_g_6)
print("times_g_7 = ", times_g_7)

print("errs_g_4 = ", errs_g_4)
print("errs_g_5 = ", errs_g_5)
print("errs_g_6 = ", errs_g_6)
print("errs_g_7 = ", errs_g_7)

print("n_basis_g_4 = ", n_basis_g_4)
print("n_basis_g_5 = ", n_basis_g_5)
print("n_basis_g_6 = ", n_basis_g_6)
print("n_basis_g_7 = ", n_basis_g_7)


print("n_mesh_lspg_4 = ", n_mesh_lspg_4)
print("n_mesh_lspg_5 = ", n_mesh_lspg_5)
print("n_mesh_lspg_6 = ", n_mesh_lspg_6)
print("n_mesh_lspg_7 = ", n_mesh_lspg_7)

print("times_lspg_4 = ", times_lspg_4)
print("times_lspg_5 = ", times_lspg_5)
print("times_lspg_6 = ", times_lspg_6)
print("times_lspg_7 = ", times_lspg_7)

print("errs_lspg_4 = ", errs_lspg_4)
print("errs_lspg_5 = ", errs_lspg_5)
print("errs_lspg_6 = ", errs_lspg_6)
print("errs_lspg_7 = ", errs_lspg_7)

print("n_basis_lspg_4 = ", n_basis_lspg_4)
print("n_basis_lspg_5 = ", n_basis_lspg_5)
print("n_basis_lspg_6 = ", n_basis_lspg_6)
print("n_basis_lspg_7 = ", n_basis_lspg_7)

"""
channel_names = [
    r"$^{1}S_{0}$",
    r"$^{3}P_{0}$",
    r"$^{1}P_{1}$",
    r"$^{3}P_{1}$",
    r"$^{1}D_{2}$",
    r"$^{3}D_{2}$",
    r"$^{1}F_{3}$",
    r"$^{3}F_{3}$",
    r"$^{1}G_{4}$",
    r"$^{3}G_{4}$",
    r"$^{1}H_{5}$",
    r"$^{3}H_{5}$",
    r"$^{1}I_{6}$",
    r"$^{3}I_{6}$",
    r"$^{3}S_{1}\text{–}^{3}D_{1}$",
    r"$^{3}P_{2}\text{–}^{3}F_{2}$",
    r"$^{3}D_{3}\text{–}^{3}G_{3}$",
    r"$^{3}F_{4}\text{–}^{3}H_{4}$",
    r"$^{3}G_{5}\text{–}^{3}I_{5}$",
    r"$^{3}H_{6}\text{–}^{3}K_{6}$",
]
"""

channel_names = [
    "1S0",
    "3P0",
    "1P1",
    "3P1",
    "1D2",
    "3D2",
    "1F3",
    "3F3",
    "1G4",
    "3G4",
    "1H5",
    "3H5",
    "1I6",
    "3I6",
    "3S1-3D1",
    "3P2-3F2",
    "3D3-3G3",
    "3F4-3H4",
    "3G5-3I5",
    "3H6-3K6",
]

energies = ['10 MeV', '50 MeV', '100 MeV', '200 MeV']
markers = ['s', 'o', '^', 'x']

##############################################################################

fig, ax = plt.subplots(1, 1, figsize=(col_width, col_width))

for k in n_basis_g_6.keys():

    channels = list( n_basis_g_6[k].keys() )
    n_basis = list( n_basis_g_6[k].values() )

    ax.scatter(channels, n_basis, label=energies[k], marker=markers[k], s=50)

ax.axhline(2, color='k', label='Initial Basis Size', zorder=-1)
x = list(range(len(channel_names)))
ax.set_xticks(x, channel_names, rotation=45)
ax.tick_params(axis='x', which='minor', bottom=False, top=False)


plt.ylabel("Basis Size")
#plt.xlabel("Channel")
plt.legend()
plt.savefig(r"basis_size_vs_channel_various_energies_grom.pdf", format='pdf')
#plt.show()
plt.close()

##############################################################################

fig, ax = plt.subplots(1, 1, figsize=(col_width, col_width))

for k in n_basis_lspg_6.keys():

    channels = list( n_basis_lspg_6[k].keys() )
    n_basis = list( n_basis_lspg_6[k].values() )

    ax.scatter(channels, n_basis, label=energies[k], marker=markers[k], s=50)

ax.axhline(2, color='k', label='Initial Basis Size', zorder=-1)
x = list(range(len(channel_names)))
ax.set_xticks(x, channel_names, rotation=90)
ax.tick_params(axis='x', which='minor', bottom=False, top=False)


ax.set_ylabel("Basis Size")
ax.set_xlabel("Channel")
plt.legend()
plt.savefig(r"basis_size_vs_channel_various_energies_lspgrom.pdf", format='pdf')
#plt.show()
plt.close()

##############################################################################


markers = ['s', 'o', '^', 'x']

fig, ax = plt.subplots(1, 2, figsize=(text_width, col_width), sharex=True, sharey=True)


k = 3
channels = list( n_basis_g_4[k].keys() )
n_basis = list( n_basis_g_4[k].values() )
ax[1].scatter(channels, n_basis, label=r'$\alpha = 10^{-4}$', marker='s', s=50, color='C0')

channels = list( n_basis_g_5[k].keys() )
n_basis = list( n_basis_g_5[k].values() )
ax[1].scatter(channels, n_basis, label=r'$\alpha = 10^{-5}$', marker='o', s=50, color='C1')

channels = list( n_basis_g_6[k].keys() )
n_basis = list( n_basis_g_6[k].values() )
ax[1].scatter(channels, n_basis, label=r'$\alpha = 10^{-6}$', marker='^', s=50, color='C2')

channels = list( n_basis_g_7[k].keys() )
n_basis = list( n_basis_g_7[k].values() )
ax[1].scatter(channels, n_basis, label=r'$\alpha = 10^{-7}$', marker='x', s=50, color='C3')


ax[1].axhline(2, color='k', label='Initial Basis Size', zorder=-1)



##############################################################################

k = 0
channels = list( n_basis_g_4[k].keys() )
n_basis = list( n_basis_g_4[k].values() )
ax[0].scatter(channels, n_basis, label=r'$\alpha = 10^{-4}$', marker='s', s=50, color='C0')

channels = list( n_basis_g_5[k].keys() )
n_basis = list( n_basis_g_5[k].values() )
ax[0].scatter(channels, n_basis, label=r'$\alpha = 10^{-5}$', marker='o', s=50, color='C1')

channels = list( n_basis_g_6[k].keys() )
n_basis = list( n_basis_g_6[k].values() )
ax[0].scatter(channels, n_basis, label=r'$\alpha = 10^{-6}$', marker='^', s=50, color='C2')

channels = list( n_basis_g_7[k].keys() )
n_basis = list( n_basis_g_7[k].values() )
ax[0].scatter(channels, n_basis, label=r'$\alpha = 10^{-7}$', marker='x', s=50, color='C3')


ax[0].axhline(2, color='k', label='Initial Basis Size', zorder=-1)



x = list(range(len(channel_names)))
ax[0].set_xticks(x, channel_names, rotation=90)
ax[1].set_xticks(x, channel_names, rotation=90)
ax[0].tick_params(axis='x', which='minor', bottom=False, top=False)
ax[1].tick_params(axis='x', which='minor', bottom=False, top=False)

ax[0].set_title(r"$E_{\mathrm{lab}} = $ 10 MeV")
ax[1].set_title(r"$E_{\mathrm{lab}} = $ 200 MeV")

ax[0].set_ylabel("Basis Size")
#ax[0].set_xlabel("Channel")
#ax[1].set_xlabel("Channel")
ax[0].legend()
plt.savefig(r"basis_size_vs_channel_10MeV_200MeV_various_tol_grom.pdf", format='pdf')
#plt.show()
plt.close()

##############################################################################

def get_speedup(times_rom, errs_rom):
    speedup = times_solver / times_rom
    rel2 = (errs_solver / times_solver)**2 + (errs_rom / times_rom)**2
    return speedup, speedup * np.sqrt(rel2)
    


def scaling_band(x, y, yerr, x_mesh, nsigma=1.0, ):


    x = np.asarray(x)
    y = np.asarray(y)
    yerr = np.asarray(yerr)

    # Transform to log-log
    logx = np.log(x)
    logy = np.log(y)
    sigma_logy = yerr / y  # propagated uncertainty

    # Weighted linear regression: log y = a + f log x
    A = np.vstack([np.ones_like(logx), logx]).T
    W = np.diag(1.0 / sigma_logy**2)

    ATA = A.T @ W @ A
    ATy = A.T @ W @ logy
    params = np.linalg.solve(ATA, ATy)
    cov = np.linalg.inv(ATA)

    a, f = params
    da, df = np.sqrt(np.diag(cov))
    
    print(f"scaling: {f} +- {df}")

    # Predictions in log space
    logx_mesh = np.log(x_mesh)
    logy_fit = a + f * logx_mesh

    # Variance of predicted log y at each x
    var_logy = (
        cov[0, 0]
        + 2 * cov[0, 1] * logx_mesh
        + cov[1, 1] * logx_mesh**2
    )
    sigma_pred = np.sqrt(var_logy)

    # Lower and upper in log space
    logy_lo = logy_fit - nsigma * sigma_pred
    logy_hi = logy_fit + nsigma * sigma_pred

    # Convert back to linear space
    y_fit = np.exp(logy_fit)
    y_lo = np.exp(logy_lo)
    y_hi = np.exp(logy_hi)

    return f, df, y_fit, y_lo, y_hi

lims = (15, 180)
x_mesh = np.linspace(*lims)

S_g_4, dS_g_4 = get_speedup(times_g_4, errs_g_4)
S_g_5, dS_g_5 = get_speedup(times_g_5, errs_g_5)
S_g_6, dS_g_6 = get_speedup(times_g_6, errs_g_6)
S_g_7, dS_g_7 = get_speedup(times_g_7, errs_g_7)

S_lspg_4, dS_lspg_4 = get_speedup(times_lspg_4, errs_lspg_4)
S_lspg_5, dS_lspg_5 = get_speedup(times_lspg_5, errs_lspg_5)
S_lspg_6, dS_lspg_6 = get_speedup(times_lspg_6, errs_lspg_6)
S_lspg_7, dS_lspg_7 = get_speedup(times_lspg_7, errs_lspg_7)


print("S_g_4", S_g_4)
print("S_g_5", S_g_5)
print("S_g_6", S_g_6)
print("S_g_7", S_g_7)

print("dS_g_4", dS_g_4)
print("dS_g_5", dS_g_5)
print("dS_g_6", dS_g_6)
print("dS_g_7", dS_g_7)

print("S_lspg_4", S_lspg_4)
print("S_lspg_5", S_lspg_5)
print("S_lspg_6", S_lspg_6)
print("S_lspg_7", S_lspg_7)

print("dS_lspg_4", dS_lspg_4)
print("dS_lspg_5", dS_lspg_5)
print("dS_lspg_6", dS_lspg_6)
print("dS_lspg_7", dS_lspg_7)



fig, ax = plt.subplots(1, 1, figsize=(col_width, col_width))



ax.errorbar(n_mesh_solver, S_g_4, yerr=dS_g_4, marker='s', linestyle='none', elinewidth=1, markersize=6, markerfacecolor='white', markeredgecolor='C0', label=r'G,  $\alpha = 10^{-4}$', zorder=0)
f, df, y_fit, y_lo, y_hi = scaling_band(n_mesh_solver, S_g_4, dS_g_4, x_mesh)
ax.plot(x_mesh, y_fit, color='C0')


ax.errorbar(n_mesh_solver, S_g_5, yerr=dS_g_5, marker='s', linestyle='none', elinewidth=1, markersize=6, markerfacecolor='white', markeredgecolor='C1', label=r'G, $\alpha = 10^{-5}$', zorder=0)
f, df, y_fit, y_lo, y_hi = scaling_band(n_mesh_solver, S_g_5, dS_g_5, x_mesh)
ax.plot(x_mesh, y_fit, color='C1')


ax.errorbar(n_mesh_solver, S_g_6, yerr=dS_g_6, marker='s', linestyle='none', elinewidth=1, markersize=6, markerfacecolor='white', markeredgecolor='C2', label=r'G, $\alpha = 10^{-6}$', zorder=0)
f, df, y_fit, y_lo, y_hi = scaling_band(n_mesh_solver, S_g_6, dS_g_6, x_mesh)
ax.plot(x_mesh, y_fit, color='C2')


ax.errorbar(n_mesh_solver, S_g_7, yerr=dS_g_7, marker='s', linestyle='none', elinewidth=1, markersize=6, markerfacecolor='white', markeredgecolor='C3', label=r'G, $\alpha = 10^{-7}$', zorder=0)
f, df, y_fit, y_lo, y_hi = scaling_band(n_mesh_solver, S_g_7, dS_g_7, x_mesh)
ax.plot(x_mesh, y_fit, color='C3')

ax.errorbar(n_mesh_solver, S_lspg_4, yerr=dS_lspg_4, marker='o', linestyle='none', elinewidth=1, markersize=5, color='C0', label=r'LSPG, $\alpha = 10^{-4}$', zorder=2)
f, df, y_fit, y_lo, y_hi = scaling_band(n_mesh_solver, S_lspg_4, dS_lspg_4, x_mesh)
ax.plot(x_mesh, y_fit, color='C0', linestyle='dashed')


ax.errorbar(n_mesh_solver, S_lspg_5, yerr=dS_lspg_5, marker='o', linestyle='none', elinewidth=1, markersize=5, color='C1', label=r'LSPG, $\alpha = 10^{-5}$', zorder=2)
f, df, y_fit, y_lo, y_hi = scaling_band(n_mesh_solver, S_lspg_5, dS_lspg_5, x_mesh)
ax.plot(x_mesh, y_fit, color='C1', linestyle='dashed')


ax.errorbar(n_mesh_solver, S_lspg_6, yerr=dS_lspg_6, marker='o', linestyle='none', elinewidth=1, markersize=5, color='C2', label=r'LSPG, $\alpha = 10^{-6}$', zorder=2)
f, df, y_fit, y_lo, y_hi = scaling_band(n_mesh_solver, S_lspg_6, dS_lspg_6, x_mesh)
ax.plot(x_mesh, y_fit, color='C2', linestyle='dashed')


ax.errorbar(n_mesh_solver, S_lspg_7, yerr=dS_lspg_7, marker='o', linestyle='none', elinewidth=1, markersize=5, color='C3', label=r'LSPG, $\alpha = 10^{-7}$', zorder=2)
f, df, y_fit, y_lo, y_hi = scaling_band(n_mesh_solver, S_lspg_7, dS_lspg_7, x_mesh)
ax.plot(x_mesh, y_fit, color='C3', linestyle='dashed')




ax.set_xlim(*lims)
ax.set_yscale('log')
ax.set_xscale('log')
plt.legend(fontsize=7.5, loc='lower center', facecolor='white', ncol=2)

ax.set_ylim(0.07, 400)
ax.text(18, 150, r"$S_{\mathrm{G}} \propto N^{2.09(6)}$", fontsize=10)
ax.text(18, 80, r"$S_{\mathrm{LSPG}} \propto N^{2.07(7)}$", fontsize=10)

ax.set_ylabel(r'Speedup $S$')
ax.set_xlabel(r'Momentum Grid Size $N$')

plt.savefig(r"speedup_scaling.pdf", format="pdf")
#plt.show()
plt.close()


##############################################################################

fig, ax = plt.subplots(1, 1, figsize=(col_width, col_width))

tols = np.array([1e-7, 1e-6, 1e-5, 1e-4])
print(tols)

n = 3

lims = (7e-8, 1.5e-4)
#lims = (-7, -4)
x_mesh = np.linspace(*lims)

for i, n in enumerate([0, 1, 3, 5]):

    S_g_tols = np.array([S_g_7[n], S_g_6[n], S_g_5[n], S_g_4[n]])
    dS_g_tols = np.array([dS_g_7[n], dS_g_6[n], dS_g_5[n], dS_g_4[n]])
    S_lspg_tols = np.array([S_lspg_7[n], S_lspg_6[n], S_lspg_5[n], S_lspg_4[n]])
    dS_lspg_tols = np.array([dS_lspg_7[n], dS_lspg_6[n], dS_lspg_5[n], dS_lspg_4[n]])
    
    print(n_mesh_solver[n])

    ax.errorbar(tols, S_g_tols, marker='s', linestyle='none', elinewidth=1, markersize=6, markerfacecolor='white', markeredgecolor='C'+str(i), label='G, $N=$ '+str(n_mesh_solver[n]))
    f, df, y_fit, y_lo, y_hi = scaling_band(tols, S_g_tols, dS_g_tols, x_mesh)
    ax.plot(x_mesh, y_fit, color='C'+str(i))

for i, n in enumerate([0, 1, 3, 5]):

    S_g_tols = np.array([S_g_7[n], S_g_6[n], S_g_5[n], S_g_4[n]])
    dS_g_tols = np.array([dS_g_7[n], dS_g_6[n], dS_g_5[n], dS_g_4[n]])
    S_lspg_tols = np.array([S_lspg_7[n], S_lspg_6[n], S_lspg_5[n], S_lspg_4[n]])
    dS_lspg_tols = np.array([dS_lspg_7[n], dS_lspg_6[n], dS_lspg_5[n], dS_lspg_4[n]])
    
    print(n_mesh_solver[n])


    ax.errorbar(tols, S_lspg_tols, marker='o', linestyle='none', elinewidth=1, markersize=5, color='C'+str(i), label='LSPG, $N=$ '+str(n_mesh_solver[n]))
    f, df, y_fit, y_lo, y_hi = scaling_band(tols, S_lspg_tols, dS_lspg_tols, x_mesh)
    ax.plot(x_mesh, y_fit, color='C'+str(i), linestyle='dashed')

ax.set_xlim(*lims)
ax.set_yscale('log')
ax.set_xscale('log')
#plt.legend(fontsize=8, loc='lower right', facecolor='white')


ax.text(5e-6, 1.2, r"$S_{N=20} \propto \alpha^{0.13(4)}$", fontsize=10, color='C0')
ax.text(5e-6, 5, r"$S_{N=40} \propto \alpha^{0.144(5)}$", fontsize=10, color='C1')
ax.text(5e-6, 29, r"$S_{N=80} \propto \alpha^{0.165(4)}$", fontsize=10, color='C2')
ax.text(1e-7, 60, r"$S_{N=120} \propto \alpha^{0.189(6)}$", fontsize=10, color='C3')

ax.set_ylabel(r'Speedup $S$')
ax.set_xlabel(r'Requested ROM Tolerance $\alpha$')
plt.legend(loc='upper center', ncol=2, fontsize=7.5)
plt.ylim(0.7, 600)
plt.savefig(r"speedup_tol.pdf", format="pdf")
#plt.show()

