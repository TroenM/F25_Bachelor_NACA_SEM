import numpy as np
import firedrake as fd

#### Running from F25_Bachelor_NACA_SEM
import os
import sys
sys.path.append(os.getcwd())

from PoissonSolver.PoissonCLS.poisson_solver import PoissonSolver
from Meshing.mesh_library import *

class PotentialFlowSolver:
    """
    Class for solving potential flow around an airfoil using kutta condition

    """

    def __init__(self, airfoil : str = "0012", P : int = 1, alpha : float = 0, V_inf : float = 1.0, **kwargs):
        self.airfoil = airfoil
        self.P = P
        self.V_inf = V_inf
        self.kwargs = kwargs
        
        self.xlim = self.kwargs.get("xlim", [-7, 13])
        self.ylim = self.kwargs.get("ylim", [-2, 1])

        self.mesh = meshio_to_fd(naca_mesh(self.airfoil, self.alpha, self.xlim, self.ylim))




