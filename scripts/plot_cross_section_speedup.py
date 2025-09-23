import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["STIX"],       # or "Times New Roman"
    "mathtext.fontset": "stix",   # ensures math matches the text
    "font.size": 12
    #"axes.titlesize": 12, # title
    #"axes.labelsize": 11, # axis labels
    #"xtick.labelsize": 9,
    #"ytick.labelsize": 9,
    #"legend.fontsize": 9
})



n_mesh = np.array([20, 40, 60, 80, 120, 160])

speedup4 = np.array([
    [5.118990756233744, 1.5301015303527705],
    [37.973511706491166, 11.011201380553965],
    [145.98343970956168, 42.63071830652379],
    [280.3986771738497, 78.12451330488844],
    [473.9243513766383, 138.12470785246597],
    [838.1746863579166, 261.95281847625984]
]).T

speedup2 = np.array([
    [5.754578014111499, 1.7451614789652932],
    [37.091925209660424, 10.472347180745555],
    [125.90295995369108, 37.12752161653226],
    [299.9465970033432, 85.7750013711419],
    [515.8403554379067, 144.24932621497834],
    [829.8770632521138, 196.9058146126312]
]).T

x_mesh = np.linspace(10, 300, 10)


fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=300)
ax.tick_params(direction="in", top=True, right=True)  # ticks on all sides
ax.minorticks_on()

for speedup, Jmax, color in zip([speedup2, speedup4], [2, 4], ['C0', 'C1']):

    x = n_mesh
    y = speedup[0]
    yerr = speedup[1]

    plt.errorbar(x, y, yerr=yerr, marker='o', label=r'$J_{max}=$ '+str(Jmax), linewidth=0, elinewidth=1, markersize=4)



x = np.concatenate((n_mesh, n_mesh))
y = np.concatenate((speedup2[0], speedup4[0]))
yerr = np.concatenate((speedup2[1], speedup4[1]))

logx = np.log(x)
logy = np.log(y)
logyerr = yerr / y  # propagate to log-space

def line(x, m, logA):
    return m*x + logA

popt, pcov = curve_fit(line, logx, logy, sigma=logyerr, absolute_sigma=True)

m, logA = popt
dm, dlogA = np.sqrt(np.diag(pcov))
A = np.exp(logA)
dA = A * dlogA

print(f"Slope (m): {m:.3f} ± {dm:.3f}")
print(f"Prefactor (A): {A:.3f} ± {dA:.3f}")

plt.fill_between(x_mesh, (A-dA) * x_mesh ** (m-dm), (A+dA) * x_mesh ** (m+dm), alpha=0.2, color='k', label=r'$\pm 1 \sigma$ band')

plt.yscale('log')
plt.xscale('log')

plt.text(60, 2, r'$\propto N^{2.43(12)}$', size='x-large')


ax.set_ylabel('Emulator Speedup')
ax.set_xlabel('Number of Integration Points $N$')
#plt.fill_between([1], [1], [1], label=r'$1\sigma$ confidence band', color='k', alpha=0.2)
ax.set_xlim(x_mesh[0], x_mesh[-1])
plt.legend()
plt.xticks([10, 100], [r"$10^1$", r"$10^2$"])

plt.savefig("figures/cross_section_speedup.pdf", bbox_inches="tight")

