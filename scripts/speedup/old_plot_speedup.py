import numpy as np
import matplotlib.pyplot as plt



# Fixed Jmax = 6, Elab = 100 MeV
# 14 single channels, 6 coupled channels
# calculated using 100 samples after warming up
# reporting mean and std err of mean

def get_solver_data(filename):
    return np.loadtxt(filename, unpack=True, skiprows=1, delimiter=',')



def get_emulator_data(filename):

    Nqs = []
    avgs = []
    errs = []

    with open(filename, "r") as f:
        for line in f:
            if "*" in line and "***" not in line:
                split_line = line.split(",")
                
                Nq = int(split_line[0][2:])
                avg = float(split_line[1])
                err = float(split_line[2][:-2])
                
                Nqs.append(Nq)
                avgs.append(avg)
                errs.append(err)
                
            print(line)
                
    return np.array(Nqs), np.array(avgs), np.array(errs)
                
                
                
            
Nq, avg_solver, err_solver = get_solver_data("scripts/speedup/results/exact.txt")
_, avg_grom4, err_grom4 = get_emulator_data("scripts/speedup/results/greedy_grom_1e-4.txt")
_, avg_grom7, err_grom7 = get_emulator_data("scripts/speedup/results/greedy_grom_1e-7.txt")
_, avg_lspgrom4, err_lspgrom4 = get_emulator_data("scripts/speedup/results/greedy_lspgrom_1e-4.txt")
_, avg_lspgrom7, err_lspgrom7 = get_emulator_data("scripts/speedup/results/greedy_lspgrom_1e-7.txt")


speedup_grom4 = avg_solver / avg_grom4
speedup_grom7 = avg_solver / avg_grom7
speedup_lspgrom4 = avg_solver / avg_lspgrom4
speedup_lspgrom7 = avg_solver / avg_lspgrom7

print(speedup_grom4)
print(speedup_grom7)
print(speedup_lspgrom4)
print(speedup_lspgrom7)


fig, ax = plt.subplots(1, 1, figsize=(6,5))

ms = 10
mew = 2
lw = 2
ax.errorbar(Nq, speedup_grom4, marker='o', label=r'G-ROM (tol=$10^{-4}$)', markersize=ms, linewidth=lw)
ax.errorbar(Nq, speedup_grom7, marker='^', label=r'G-ROM (tol=$10^{-7}$)', markersize=ms, linewidth=lw)
ax.errorbar(Nq, speedup_lspgrom4, marker='s', markeredgewidth=mew, linestyle='dotted', label=r'LSPG-ROM (tol=$10^{-4}$)', markersize=ms, linewidth=lw)
ax.errorbar(Nq, speedup_lspgrom7, marker='x', markeredgewidth=mew, linestyle='dashed', label=r'LSPG-ROM (tol=$10^{-7}$)', markersize=ms, linewidth=lw)

ax.set_xlabel('Number of Mesh Points')
ax.set_ylabel('Speedup')
ax.set_yscale('log')
ax.set_xscale('log')
plt.legend()

plt.show()
