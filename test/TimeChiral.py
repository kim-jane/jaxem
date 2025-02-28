import jax
from jax import numpy as jnp
from jax import random, vmap
import pickle
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from jaxem import *
import numpy as np
from scipy.linalg import orth

ti = time.time()


n_devices = jax.local_device_count()
print(f"n_devices = {n_devices}")


# read inputs
config = Config('test/TimeChiral.ini')


# create tmat
tmat = TMatrix(config)

# define boundaries
LECs_best = tmat.pot.LECs

LECs_lbd = 0.0 * LECs_best
LECs_lbd = LECs_lbd.at[0].set(LECs_best[0])

LECs_ubd = 2.0 * LECs_best
LECs_ubd = LECs_ubd.at[0].set(LECs_best[0])

# POD GROM
tmat.train_POD_GROM(LECs_lbd, LECs_ubd)



# greedy GROM
tmat.train_greedy_GROM(LECs_lbd, LECs_ubd, o1=8, o2=9)
