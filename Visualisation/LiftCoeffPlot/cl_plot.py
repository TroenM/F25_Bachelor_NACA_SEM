import numpy as np
import firedrake as fd
import shutil
import os
import sys
from time import time

#### Running from F25_Bachelor_NACA_SEM ####
sys.path.append(os.getcwd())

from Potential_flow_solver.PotentialFlowSolverCLS import PotentialFlowSolver
os.chdir("./Visualisation")

nasa_cl = np.loadtxt("nasa_cl.txt")
circular_cl = np.loadtxt("circular_cl.txt")
eliptic_cl = np.loadtxt("eliptic_cl.txt")
plt.plot()