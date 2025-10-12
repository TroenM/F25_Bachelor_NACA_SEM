import os;
os.environ["OMP_NUM_THREADS"] = "1"
os.system('cls||clear')

from MeshEssentials import *
import firedrake as fd
from firedrake.pyplot import tripcolor
import numpy as np
import matplotlib.pyplot as plt
from time import time
import copy

"""
IMPORTANT: The boundaries should be indexed as follows:
1: Inflow
2: Outflow
3: Bed
4: Free Surface
5: Airfoil
"""

hypParams = {
    "P": 1, # Polynomial degree
    "V_inf": fd.as_vector((1.0, 0.0)), # Free stream velocity
    "rho": 1.225 # Density of air [kg/m^3]
}

meshSettings = {
    "airfoilNumber": "0012", # NACA airfoil number
    "alpha_deg": 3, # Angle of attack in degrees
    "centerOfAirfoil": (0.0, 0), # Center of airfoil (x,y)
    "circle": True, # Whether to use a circular or elliptical vortex
    "xlim": (-7, 13), # x-limits of the domain
    "ylim": (-4, 2), # y-limits of the domain
    "nIn": 20, # Number of external nodes on inlet boundary 
    "nOut": 20, # Number of external nodes on outlet boundary
    "nBed": 50, # Number of external nodes on bed boundary
    "nFS": 300, # Number of external nodes on free surface boundary
    "nAirfoil": 100 # Number of external nodes on airfoil boundary
}

solverSettings = {
    "maxItKutta": 50,
    "tolKutta": 1e-6,
    "maxItFreeSurface": 50,
    "tolFreeSurface": 1e-6,
    "c0": 7, # Initial guess for the adaptive stepsize controller for Gamma
    "dt": 1e-3 # Time step for free surface update
}

outputSettings = {
    "outputPath": "./Results/",
    "writeKutta": True, # Whether to write output for each Kutta iteration
    "writeFreeSurface": True, # Whether to write output for each free surface iteration
    "outputIntervalKutta": 1, # Output interval in time steps
    "outputIntervalFS": 25, # Output interval in free surface time steps
}

class FSSolver:

    #==================================================================#
    #======================== Initialization ==========================#
    #==================================================================#

    def __init__(self, hypParams: dict, meshSettings: dict, solverSettings: dict, outputSettings: dict) -> None:
        # Hyperparameters
        time_init = time()
        self.P = hypParams["P"]
        self.V_inf = hypParams["V_inf"]
        self.rho = hypParams["rho"]

        # Mesh parameters
        self.airfoilNumber = meshSettings["airfoilNumber"]
        self.centerOfAirfoil = meshSettings["centerOfAirfoil"]
        self.alpha = np.deg2rad(meshSettings["alpha_deg"])
        self.circle = meshSettings["circle"]
        self.xlim = meshSettings["xlim"]
        self.ylim = meshSettings["ylim"]
        self.nIn = meshSettings["nIn"]
        self.nOut = meshSettings["nOut"]
        self.nBed = meshSettings["nBed"]
        self.nFS = meshSettings["nFS"]
        self.nAirfoil = meshSettings["nAirfoil"]

        # Computed mesh parameters
        self.mesh = naca_mesh(self.airfoilNumber, np.rad2deg(self.alpha), self.xlim, self.ylim, center_of_airfoil=self.centerOfAirfoil,
                              n_in=self.nIn, n_out=self.nOut, n_bed=self.nBed, n_fs=self.nFS, n_airfoil=self.nAirfoil)
        self.a = 1
        self.b = 1 if self.circle else int(self.airfoilNumber[2:])/100

        self.LE, self.TE, self.vPerp, self.pointAtTE = self.__findAirfoilDetails__()

        # Solver parameters
        self.maxItKutta = solverSettings["maxItKutta"]
        self.tolKutta = solverSettings["tolKutta"]
        self.maxItFreeSurface = solverSettings["maxItFreeSurface"]
        self.tolFreeSurface = solverSettings["tolFreeSurface"]
        self.c0 = solverSettings["c0"]
        self.Gammas = []
        self.dt = solverSettings["dt"]
        

        # Function spaces
        self.V = fd.FunctionSpace(self.mesh, "CG", self.P)
        self.W = fd.VectorFunctionSpace(self.mesh, "CG", self.P)

        fs_indecies = self.V.boundary_nodes(4)
        self.coordsFS = (fd.Function(self.W).interpolate(self.mesh.coordinates).dat.data)[fs_indecies,:]

        # Output parameters
        self.outputPath = outputSettings["outputPath"]
        self.writeKutta = outputSettings["writeKutta"]
        self.writeFreeSurface = outputSettings["writeFreeSurface"]
        self.outputIntervalKutta = outputSettings["outputIntervalKutta"]
        self.outputIntervalFS = outputSettings["outputIntervalFS"]

        print("Initialized FSSolver with:\n" + f"P={self.P}\n" + f"alpha={np.round(np.rad2deg(self.alpha), 2)} deg\n" + 
              f"V_inf={self.V_inf}\n" + f"Degrees of freedom: {self.V.dof_count}")
        print(f"Initialization time: {np.round(time() - time_init, 2)} s")
        print("-"*50 + "\n")
        return None
    

    def __normaliseVector__(self, vector : np.ndarray) -> np.ndarray:
        if type(vector) != np.ndarray or np.linalg.norm(vector) == 0:
            raise TypeError("The vector has to be a numpy array of length more than 0")
        return vector/np.linalg.norm(vector)

    def __findAirfoilDetails__(self) -> tuple: #Find the leading and trailing edge of the airfoil
        '''
        Find the leading and trailing edge of the airfoil by centering and rotating the airfoil to alpha = 0 
        and finding the min and max x-coordinates, then rotating back to the original angle of attack and shifting back to the original center.

        THIS IS ONLY NECESSARY ONCE, AT THE START OF THE SIMULATION.
        '''
        # Calculate airfoil coordinates
        naca_coords = naca_4digit(self.airfoilNumber,self.nAirfoil, self.alpha, self.centerOfAirfoil)
        # Gathering position of Leading edge, Trailing edge, 
        # the first point on the bottom surface from the trailing edge (p1) and the first on the top surface (pn)
        TE = naca_coords[0]
        LE = naca_coords[self.nAirfoil//2]
        p1 = naca_coords[1]
        pn = naca_coords[-1]

        # Calculate a normalised vector going from p1 to TE and a normalizes vector going from pn to TE
        v1 = self.__normaliseVector__(TE - p1)
        vn = self.__normaliseVector__(TE - pn)

        # Using these vectors to calculate the normalized vector that is orthorgonal 
        # to the direction of the trailing edge (vPerp)
        vPerp = self.__normaliseVector__(v1 - vn)

        # Using vPerp to find a point that is just outside the trailing edge in the direction of the trailing edge
        pointAtTE = TE + np.array([vPerp[1], -vPerp[0]])/70

        return LE, TE, vPerp, pointAtTE
    
    #=================================================================#
    #======================== Poisson Solver =========================#
    #=================================================================#

    def __poissonSolver__(self, rhs = fd.Constant(0), DBC = [], NBC = []):
        v = fd.TestFunction(self.V)
        phi = fd.TrialFunction(self.V)
        a = fd.inner(fd.grad(phi), fd.grad(v)) * fd.dx
        L = rhs*v*fd.dx

        DBCs = []
        for _ in DBC:
            bcidx, DBCfunc = _
            DBCs.append(fd.DirichletBC(self.V, DBCfunc, bcidx))
    
        for _ in NBC:
            bcidx, NBCfunc = _
            L += fd.dot(NBCfunc, fd.FacetNormal(self.mesh)) * v * fd.ds(bcidx) # Set NBC = [fd.as_vector(V_inf, 0)]*2 for far-field

        phi = fd.Function(self.V) # Consider whether phi should be an input instead.
        fd.solve(a == L, phi, bcs=DBCs)

        if len(DBCs) == 0: # Normalize phi if there are no Dirichlet BCs
            phi -= np.min(phi.dat.data)
        
        u = fd.Function(self.W).interpolate(fd.grad(phi))

        return phi, u
    
    #=================================================================#
    #======================== Kutta Condition ========================#
    #=================================================================#
    def __FBCS__(self) -> float: # Applies the FBCS-scheme discussed in the report
        # Adaptive stepsize controller for Gamma described in the report
        Gammas = self.Gammas
        c0 = copy.copy(self.c0)
        if len(Gammas) == 1:
            return Gammas[-1]/c0
        else:
            a = (Gammas[-1]/c0) / Gammas[-2]
            c0 *= (1-a)
            self.c0 = c0
            return Gammas[-1]/c0
    
    def __computeVortexStrength__(self) -> float:
        """
        Computes the vortex strength for the given iteration
        """
        alpha = self.alpha
        a = self.a
        b = self.b
        # Get the coordinates of the point just outside of the trailing edge
        p_x = PointAtTE[0]
        p_y = PointAtTE[1]

        # Translating the trailing edge coordinates to have the center at the origin
        x_t = p_x - centerOfAirfoil[0]
        y_t = p_y - centerOfAirfoil[1]

        # Rotating the trailing edge coordinates to align with "not-rotated" coordinates
        x_bar = x_t * np.cos(alpha) - y_t * np.sin(alpha)
        y_bar = x_t * np.sin(alpha) + y_t * np.cos(alpha)

        # Computing vortex at trailing edge without  the scaling factor Gamma/ellipseCircumference
        Wx = -(y_bar/b) / (x_bar**2/a + y_bar**2/b)
        Wy = (x_bar/a) / (x_bar**2/a + y_bar**2/b)

        # Rotating the vortex vector at the trailing edge clockwise by alpha,
        # in order to mimic that vectors orientation in the rotated vortex field
        Wx_rot = Wx * np.cos(-alpha) - Wy * np.sin(-alpha)
        Wy_rot = Wx * np.sin(-alpha) + Wy * np.cos(-alpha)

        # Calculating the circumference of the ellipse (scaling factor)
        ellipseCircumference = np.pi*(3*(a+b) - np.sqrt(3*(a+b)**2+4*a*b))

        # Computing the vortex strength Gamma
        Gamma = -ellipseCircumference*(vPerp[0]*VelocityAtTE[0] + vPerp[1]*VelocityAtTE[1])/(Wx_rot*vPerp[0] + Wy_rot*vPerp[1])

        return Gamma

if __name__ == "__main__":
    solver = FSSolver(hypParams, meshSettings, solverSettings, outputSettings)
