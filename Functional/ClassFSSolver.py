import os;
os.environ["OMP_NUM_THREADS"] = "1"
os.system('cls||clear')

from MeshEssentials import *
import firedrake as fd
from firedrake.pyplot import tripcolor
import numpy as np
import matplotlib.pyplot as plt
from time import time
import shutil
import sys
from numpy.lib.format import open_memmap

import os
if os.getcwd()[-21:] == 'F25_Bachelor_NACA_SEM':
    os.chdir("./Functional")
if not os.getcwd().endswith("Functional"):
    raise InterruptedError("""ClassFSSolver.py must be run from "Functional" or root folder, 
                           because Magnus is too lazy to fix relative import/export paths""")

"""
IMPORTANT: The boundaries should be indexed as follows:
1: Inflow
2: Outflow
3: Bed
4: Free Surface
5: Airfoil
"""

hypParams = {
    "P": 2, # Polynomial degree
    "V_inf": fd.as_vector((1.0, 0.0)), # Free stream velocity
    "rho": 1.225, # Density of air [kg/m^3]
    "nFS": 150,
    "FR": 0.5672,
    "continue": False
}

meshSettings = {
    "airfoilNumber": "0012",
    "alpha_deg": 5,
    "circle": True,

    "xlim": (-7,10),
    "y_bed": -4,

    "scale": 1,
    
    "h": 1.034,
    "interface_ratio": 5,
    "nAirfoil": hypParams["nFS"]//2,
    "centerOfAirfoil": (0.5,0.0),

    "nFS": hypParams["nFS"],
    "nUpperSides": "Calculated down below to make upper elemets square (if they were not triangular xD)",
    "nLowerInlet": hypParams["nFS"]//10,
    "nLowerOutlet": hypParams["nFS"]//10,
    "nBed": hypParams["nFS"]//2,
    "test": True
    }


def calculateNUpperSides(meshSettings):
    nFS = meshSettings["nFS"]
    xlim = meshSettings["xlim"]
    h = meshSettings["h"]
    meshSettings["nUpperSides"] =  int( nFS/(xlim[1]-xlim[0]) * h )
    return None
calculateNUpperSides(meshSettings)

solverSettings = {
    "maxItKutta": 50,
    "tolKutta": 1e-10,
    "maxItFreeSurface": 10000,
    "minItFreeSurface": 100, # Let the solver ramp up for x iterations before checking for convergence
    "tolFreeSurface": 1e-6,

    "maxItWeak1d": 2500 , # Maximum iterations for free surface SNES solver (Go crazy, this is cheap)
    "tolWeak1d": 1e-8, # Tolerance for free surface SNES solver

    "c0": 7, # Initial guess for the adaptive stepsize controller for Gamma
    "dt": 2e-2, # Time step for free surface update

    "startIteration": np.where(np.load("TestResults/arrays/residuals.npy")[:,1] == 0)[0][0]-1 if hypParams["continue"] else 0
}


outputSettings = {
    "outputPath": "./TestResults/",
    "writeKutta": True, # Whether to write output for each Kutta iteration
    "writeFreeSurface": True, # Whether to write output for each free surface iteration
    "outputIntervalKutta": 1, # Output interval in time steps
    "outputIntervalFS": 100, # Output interval in free surface time steps
}
deleteLines = False

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
        self.centerOfVortex = meshSettings.get("centerOfVortex", self.centerOfAirfoil) # Default to centerOfAirfoil if not provided
        self.alpha = np.deg2rad(meshSettings["alpha_deg"])
        self.circle = meshSettings["circle"]
        self.xlim = meshSettings["xlim"]
        # self.ylim = meshSettings["ylim"]
        self.nUpperSides = meshSettings["nUpperSides"]

        self.yBed = meshSettings["nBed"]
        self.nAirfoil = meshSettings["nAirfoil"]

        # Computed mesh parameters
        ### Choose either uniform mesh on top or original mesh
        self.mesh, self.yInterface, self.ylim = createFSMesh(self.airfoilNumber, np.rad2deg(self.alpha), meshSettings)
        #self.mesh, self.yInterface, self.ylim = naca_mesh(self.airfoilNumber, np.rad2deg(self.alpha), meshSettings)
        self.a = 1
        self.b = 1 if self.circle else int(self.airfoilNumber[2:])/100

        self.LE, self.TE, self.vPerp, self.pointAtTE = self.__findAirfoilDetails__()

        # Solver parameters
        self.maxItKutta = solverSettings["maxItKutta"]
        self.tolKutta = solverSettings["tolKutta"]
        self.maxItFreeSurface = solverSettings["maxItFreeSurface"]
        self.minItFreeSurface = solverSettings["minItFreeSurface"]
        self.tolFreeSurface = solverSettings["tolFreeSurface"]
        self.tolWeak1d = solverSettings["tolWeak1d"]
        self.maxItWeak1d = solverSettings["maxItWeak1d"]

        self.c0 = solverSettings["c0"]
        self.Gammas = []
        # self.dt = solverSettings["dt"]

        # Function spaces
        self.V = fd.FunctionSpace(self.mesh, "CG", self.P)
        self.V1 = fd.FunctionSpace(self.mesh, "CG", 1)
        self.W = fd.VectorFunctionSpace(self.mesh, "CG", self.P)
        self.W1 = fd.VectorFunctionSpace(self.mesh, "CG", 1)

        self.fSIndecies = self.W1.boundary_nodes(4) 
        self.coordsFS = (fd.Function(self.W1).interpolate(self.mesh.coordinates).dat.data)[self.fSIndecies,:]
        # Define 1D mesh along free surface
        self.fsMesh = fd.IntervalMesh(len(self.coordsFS)-1, *self.xlim)
        # Ensure nodes match the x-coordinates of free surface variables
        self.fsMesh.coordinates.dat.data[:] = self.coordsFS[:,0]

        self.W1FS = fd.VectorFunctionSpace(self.fsMesh, "CG", 1)
        self.V1FS = fd.FunctionSpace(self.fsMesh, "CG", 1)

        self.__gatherPointsAndDefineEvaluators__()

        # Computing dt idea from Simone Minniti
        sortedFSx = np.sort(np.copy(self.coordsFS[:,0]))
        diffFSx =np.diff(sortedFSx)
        dxx = np.min(diffFSx)
        self.dt = 0.1 * dxx/np.sqrt(float(self.V_inf[0]**2) + float(self.V_inf[1]**2))
        self.dt_fd = fd.Constant(self.dt)

        self.FR = hypParams["FR"]
        self.g = (self.V_inf[0]**2+self.V_inf[1]**2)/self.FR**2

        # For vortex
        self.vortex = fd.Function(self.W)
        self.alpha_fd = fd.Constant(self.alpha)
        self.a_fd = fd.Constant(self.a)
        self.b_fd = fd.Constant(self.b)
        self.fd_x, self.fd_y = fd.SpatialCoordinate(self.mesh)
        # Output parameters
        self.outputPath = outputSettings["outputPath"]
        self.writeKutta = outputSettings["writeKutta"]
        self.writeFreeSurface = outputSettings["writeFreeSurface"]
        self.outputIntervalKutta = outputSettings["outputIntervalKutta"]
        self.outputIntervalFS = outputSettings["outputIntervalFS"]

        self.startIteration = solverSettings.get("startIteration", 0)
        

        arrays_dir = os.path.join(self.outputPath, "arrays")
        os.makedirs(arrays_dir, exist_ok=True)

        n_steps = self.maxItFreeSurface + 1
        n_fs    = len(self.coordsFS)

        if not self.startIteration:
            self.etas = open_memmap(os.path.join(arrays_dir, "eta.npy"),
                                    mode="w+",
                                    dtype="float64",
                                    shape=(n_steps, n_fs))
            self.phis = open_memmap(os.path.join(arrays_dir, "phiTilde.npy"),
                                    mode="w+",
                                    dtype="float64",
                                    shape=(n_steps, n_fs))
            self.ws   = open_memmap(os.path.join(arrays_dir, "ws.npy"),
                                    mode="w+",
                                    dtype="float64",
                                    shape=(n_steps, n_fs))
            self.coordsFS_array = open_memmap(os.path.join(arrays_dir, "coordsFS.npy"),
                                            mode="w+",
                                            dtype="float64",
                                            shape=(n_steps, n_fs))
            self.residual_array = open_memmap(os.path.join(arrays_dir, "residuals.npy"),
                                            mode="w+",
                                            dtype="float64",
                                            shape=(n_steps, 2))
        else:
            # open existing as memmaps from arrays you can both read+write
            self.etas = np.load(os.path.join(arrays_dir, "eta.npy"), mmap_mode="r+")
            self.phis = np.load(os.path.join(arrays_dir, "phiTilde.npy"), mmap_mode="r+")
            self.ws   = np.load(os.path.join(arrays_dir, "ws.npy"), mmap_mode="r+")
            self.coordsFS_array = np.load(os.path.join(arrays_dir, "coordsFS.npy"), mmap_mode="r+")
            self.residual_array = np.load(os.path.join(arrays_dir, "residuals.npy"), mmap_mode="r+")


        print("Initialized FSSolver with:\n" + f"P={self.P}\n" + 
              f"alpha={np.round(np.rad2deg(self.alpha), 2)} deg\n" + 
              f"V_inf={self.V_inf}\n" + 
              f"Degrees of freedom: {self.V.dof_count}\n" + 
              f"dt: {self.dt}\n" + 
              f"ylim: {self.ylim}\n")
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
        naca_coords = naca_4digit(self.airfoilNumber,self.nAirfoil, np.rad2deg(self.alpha), self.centerOfAirfoil)
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

        if self.circle:
            self.centerOfVortex = np.array(self.centerOfVortex)
            # self.centerOfVortex -= np.array([vPerp[1], -vPerp[0]])/4
        
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
            L += fd.dot(NBCfunc, fd.FacetNormal(self.mesh)) * v * fd.ds(bcidx) # Set NBC = [fd.as_vector(V_inf, 0)] for far-field

        phi = fd.Function(self.V) # Consider whether phi should be an input instead.
        
        if len(DBCs) == 0:
            nullspace = fd.VectorSpaceBasis(constant=True, comm=self.V.mesh().comm)
            fd.solve(a == L, phi, bcs=DBCs, nullspace=nullspace)
            # Normalize phi such that upper left corner is 0
            phi -= fd.Constant(self.upperLeftEvaluator(phi)[0])
        else:
            fd.solve(a == L, phi, bcs=DBCs)

        
        
        u = fd.Function(self.W, name = "Velocity").interpolate(fd.grad(phi))
        return phi, u
    
    #=================================================================#
    #======================== Kutta Condition ========================#
    #=================================================================#
    def __FBCS__(self, Gamma) -> float: # Applies the FBCS-scheme discussed in the report
        # Adaptive stepsize controller for Gamma described in the report
        Gammas = self.Gammas
        c0 = self.c0
        if not Gammas:
            return Gamma/c0
        else:
            a = (Gamma/c0) / Gammas[-1]
            self.c0 = c0 = c0*(1-a)
            return Gamma/c0
    
    def __computeVortexStrength__(self) -> float:
        """
        Computes the vortex strength for the given iteration
        """
        alpha = self.alpha
        a = self.a
        b = self.b
        # Get the coordinates of the point just outside of the trailing edge
        p_x = self.pointAtTE[0]
        p_y = self.pointAtTE[1]

        # Translating the trailing edge coordinates to have the center at the origin
        x_t = p_x - self.centerOfAirfoil[0]
        y_t = p_y - self.centerOfAirfoil[1]

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
        vPerp = self.vPerp
        velocityAtTE = self.TEevaluator(self.u)[0] # Requires that self.u is computed before calling this function
        Gamma = -ellipseCircumference*(vPerp[0]*velocityAtTE[0] + vPerp[1]*velocityAtTE[1])/(Wx_rot*vPerp[0] + Wy_rot*vPerp[1])

        return Gamma
    
    def __computeVortex__(self) -> fd.Function:
        """
        Computes the vortex field for the given vortex strength
        """
        # Define alpha, a and b as firedrake coordinates
        alpha = self.alpha_fd
        a = self.a_fd
        b = self.b_fd

        # Gather coordinates from fd mesh
        fd_x, fd_y = self.fd_x, self.fd_y

        # Translate coordinates such that they have their center in origo
        x_translated = fd_x - self.centerOfVortex[0]
        y_translated = fd_y - self.centerOfVortex[1]

        # rotate the coordinates such that they are aranged as "unrotated coordinates"
        x_bar = (x_translated) * fd.cos(alpha) - (y_translated) * fd.sin(alpha)
        y_bar = (x_translated) * fd.sin(alpha) + (y_translated) * fd.cos(alpha)

        # Calculate the approximated circumference of the ellipse (scaling factor)
        ellipseCircumference = fd.pi*(3*(a+b) - fd.sqrt(3*(a+b)**2+4*a*b))

        # Compute the unrotated elliptical vortex field onto the "unrotated" coordinates
        Gamma = self.Gammas[-1]
        u_x = -Gamma / ellipseCircumference * y_bar/b / ((x_bar/a)**2 + (y_bar/b)**2)
        u_y = Gamma / ellipseCircumference * x_bar/a / ((x_bar/a)**2 + (y_bar/b)**2)

        # Rotate the final vectors in the vortex field
        u_xFinal = u_x * fd.cos(-alpha) - u_y * fd.sin(-alpha)
        u_yFinal = u_x * fd.sin(-alpha) + u_y * fd.cos(-alpha)

        # project the final vectorfunction onto the original coordinates of the mesh
        self.vortex.project(fd.as_vector([u_xFinal, u_yFinal]))
        return None
    
    def __boundaryCorrection__(self):
        """
        Computes the necessary boundary correction to cancle out current vortex field on the boundaries.
        This is done by solving a Poisson equation with NBC = -vortex on the boundaries.
        """
        vortex = self.vortex
        NBCs = [(i, -vortex) for i in range(1,6)] 
        phiBC, uBC = self.__poissonSolver__(NBC = NBCs)

        return phiBC, uBC

    def __checkKuttaConvergence__(self, it):
        """
        Checks whether the Kutta condition has converged
        """
        velocityAtTE = self.__normaliseVector__(self.TEevaluator(self.u)[0])
        dotProductTE = np.dot(velocityAtTE, self.vPerp)
        GammaDiff = np.inf if len(self.Gammas) < 2 else abs(self.Gammas[-1] - self.Gammas[-2])
        if abs(dotProductTE) < self.tolKutta:
            lines = 13
            if deleteLines:
                sys.stdout.write("\033[F" * lines)
                for _ in range(lines):
                    sys.stdout.write("\033[2K\033[1E")
                sys.stdout.write("\033[F" * lines)
            print(
f"""{"-"*50 + "\n"}
Kutta condition applied in {it+1} iterations
Dot product at TE: {dotProductTE}
""")
            return True
        elif GammaDiff < self.tolKutta:
            print(f"Kutta condition stagnated after {it+1} iterations")
            print(f"Gamma difference: {GammaDiff}")
            print(f"Dot product at TE: {dotProductTE}")
            return True
        elif it >= self.maxItKutta-1:
            print(f"Kutta condition was not applied in {it+1} iterations")
            print(f"Dot product at TE: {dotProductTE}")
            return True
        else:
            return False
    
    def __getLiftCoefficient__(self):
        # Compute lift based on circulation given the formula in the Kutta Jacowski theorem
        Gamma = np.sum(np.array(self.Gammas))
        V_inf = np.linalg.norm(np.array(self.V_inf, dtype=float))
        lift = -Gamma * V_inf * self.rho
        lift_coeff = lift / (1/2 * self.rho * V_inf**2)
        return lift_coeff

    def __getPressureCoefficients__(self) -> fd.Function:
        # Defining the firedrake function
        pressure = fd.Function(self.V, name = "Pressure_coeff")

        # Defining pressure coefficents in all of the domain from the formula given in the report.
        pressure.interpolate(1 - (fd.sqrt(fd.dot(self.u, self.u))/self.V_inf[0]) ** 2)
        return pressure

    def __applyKuttaCondition__(self):
        """
        Applies the Kutta condition to the current velocity field
        1. Compute vortex strength
        2. Compute vortex field
        3. Add vortex field to velocity field
        4. Apply boundary correction
        5. Check convergence
        6. Repeat until convergence, stagnation or max iterations reached
        """
        t1 = time()
        # Ensure Gammas is reset
        self.Gammas = []

        if self.writeKutta:
            if os.path.exists(self.outputPath + "kuttaIterations"):
                shutil.rmtree(self.outputPath + "kuttaIterations")

            try:
                os.remove(self.outputPath + "kuttaIterations.pvd")
            except:
                pass

            outfile = fd.VTKFile(self.outputPath + "kuttaIterations.pvd")
            self.u.rename("Velocity")
        
        for it in range(self.maxItKutta):
            # Compute vortex strength and correct it using FBCS
            Gamma = self.__computeVortexStrength__()
            self.Gammas.append(self.__FBCS__(Gamma))

            # Compute vortex field
            self.__computeVortex__()
            self.u += self.vortex

            # Apply boundary correction
            phiBC, uBC = self.__boundaryCorrection__()
            self.u += uBC

            if self.writeKutta and it % self.outputIntervalKutta == 0:
                outfile.write(self.u, time = it)

            # Check convergence
            if self.__checkKuttaConvergence__(it):
                print(f"Kutta solver time: {np.round(time() - t1, 4)} s")
                print("-"*50*self.writeKutta + "\n")
                break
        return None
    
    def __BuildPoissonSolver__(self):
        v = fd.TestFunction(self.V)
        phi_trial = fd.TrialFunction(self.V)

        rhs = fd.Constant(0.0)
        n = fd.FacetNormal(self.mesh)

        a = fd.inner(fd.grad(phi_trial), fd.grad(v)) * fd.dx
        L = rhs * v * fd.dx

        # Neumann parts: V_inf can be a Constant/Function
        V_inf_fd = self.V_inf   # Constant or Function
        for bcidx in [1, 2]:
            L += fd.dot(V_inf_fd, n) * v * fd.ds(bcidx)

        bc_inlet = fd.DirichletBC(self.V, self.phiTilde2d, 4)

        problem = fd.LinearVariationalProblem(a, L, self.phi, bcs=[bc_inlet])

        self.poissonSolver = fd.LinearVariationalSolver(problem)
        self.u = fd.Function(self.W)
        return None
    
    def __BuildBCSolver__(self):
        v = fd.TestFunction(self.V)
        phi_trial = fd.TrialFunction(self.V)

        rhs = fd.Constant(0.0)
        n = fd.FacetNormal(self.mesh)

        a = fd.inner(fd.grad(phi_trial), fd.grad(v)) * fd.dx
        L = rhs * v * fd.dx

        # self.vortex is a Function on the same mesh.
        for bcidx in range(1, 6):
            L += fd.dot(-self.vortex, n) * v * fd.ds(bcidx)

        self.phiBC = fd.Function(self.V, name="phiBC")
        self.uBC   = fd.Function(self.W, name="uBC")

        problem = fd.LinearVariationalProblem(a, L, self.phiBC)

        self.BCSolver = fd.LinearVariationalSolver(problem)
        return None

    def __buildKuttaSolver__(self):
        self.__BuildPoissonSolver__()
        self.__BuildBCSolver__()
        return None
    #=================================================================#
    #======================== Free Surface Update ====================#
    #=================================================================#
    def __saveOutputPath__(self) -> None:
        if self.writeFreeSurface:
            if os.path.exists(self.outputPath + "FSIterationsContinued"):
                shutil.rmtree(self.outputPath + "FSIterationsContinued")
            
            try:
                os.remove(self.outputPath + "FSIterationsContinued.pvd")
            except:
                pass
            if not self.startIteration:
                if os.path.exists(self.outputPath + "FSIterations"):
                    shutil.rmtree(self.outputPath + "FSIterations")
                try:
                    os.remove(self.outputPath + "FSIterations.pvd")
                except:
                    pass
                outfileFS = fd.VTKFile(self.outputPath + "FSIterations.pvd")
            else:
                outfileFS = fd.VTKFile(self.outputPath + "FSIterationsContinued.pvd")
            #self.u.rename("Velocity")
            return outfileFS
    
    def __save_results__(self):
        iter = self.iter

        self.etas[iter, :]        = self.FSxEvaluator(self.eta)
        self.phis[iter, :]        = self.FSEvaluator(self.phi)
        self.ws[iter, :]          = self.wn.dat.data_ro[:]
        self.coordsFS_array[iter,:] = self.fsMesh.coordinates.dat.data_ro[:]

        if iter == 0:
            self.residual_array[iter, 0] = self.tolFreeSurface
            self.residual_array[iter, 1] = -1
        else:
            self.residual_array[iter, 0] = self.residuals
            self.residual_array[iter, 1] = self.dt * iter
        return None

    def __doInitialKuttaSolve__(self) -> None:
        '''iter is tool for testing'''
        self.phi, self.u = self.__poissonSolver__(NBC=[(i, self.V_inf) for i in [1,2]])
        self.__applyKuttaCondition__()
        return None
    
    def __doKuttaSolve__(self):
        self.poissonSolver.solve()
        self.u.interpolate(fd.grad(self.phi))

        t1 = time()
        # Ensure Gammas is reset
        self.Gammas = []

        for it in range(self.maxItKutta):
            # Compute vortex strength and correct it using FBCS
            Gamma = self.__computeVortexStrength__()
            self.Gammas.append(self.__FBCS__(Gamma))

            # Compute vortex field
            self.__computeVortex__()
            self.u.assign(self.u + self.vortex)

            # Apply boundary correction
            self.phiBC.assign(0.0)
            self.BCSolver.solve()
            self.uBC.interpolate(fd.grad(self.phiBC))
            self.u.assign(self.u + self.uBC)

            if self.__checkKuttaConvergence__(it):
                print(f"Kutta solver time: {np.round(time() - t1, 4)} s")
                print("-"*50*self.writeKutta + "\n")
                break
        return None

    def __initPhiTilde__(self) -> None:
        V1 = self.V1FS
        if not self.startIteration:
            self.phiTilde = fd.Function(V1)
            self.phiTilde_prev = fd.Function(V1)
            self.phiTilde2d = fd.Function(self.V)

            self.phiTilde.dat.data[:] = self.FSEvaluator(self.phi)
            self.phiTilde_prev.dat.data[:] = self.FSEvaluator(self.phi)
        else:
            self.phiTilde = fd.Function(V1)
            self.phiTilde_prev = fd.Function(V1)
            self.phiTilde2d = fd.Function(self.V)
            self.phi = fd.Function(self.V)

            self.phiTilde.dat.data[:] = self.phis[self.startIteration,:]
            self.phiTilde2d = self.__lift_1d_to_2d__(self.phiTilde, self.phiTilde2d)

            self.phiTilde_prev.dat.data[:] = self.phis[self.startIteration-1,:]

        self.inletValue = fd.Constant(self.phiTilde.dat.data_ro[self.coordsFS[:,0].argmin()]) # phiTilde = constant at inflow boundary
        self.upperLeftValue = fd.Constant(self.upperLeftFSEvaluator(self.phiTilde)[0])
        return None
    
    def __initEta__(self):
        V1 = self.V1FS
        if not self.startIteration:
            self.eta = fd.Function(V1).interpolate(fd.Constant(self.ylim[1]))
            self.eta2d = fd.Function(self.V).interpolate(fd.Constant(self.ylim[1]))
            self.newEta = fd.Function(V1)
            self.wn = fd.Function(V1)
        else:
            self.eta = fd.Function(V1)
            self.newEta = fd.Function(V1)
            self.wn = fd.Function(V1)

            self.eta.dat.data[:] = self.etas[self.startIteration-1,:]
            self.eta2d = None

            self.newEta.dat.data[:] = self.etas[self.startIteration,:]
            self.newEta2d = None

            self.residuals = fd.norm(self.newEta - self.eta, norm_type='l2')

            self.wn.dat.data[:] = self.ws[self.startIteration, :]
        return None

    def __lift_1d_to_2d__(self, u1D, u2D):
        """
        Lift a scalar 1D Firedrake Function u1D(x) to a 2D Function on self.mesh
        by defining u2D(x,y) := u1D(x).
        """
        u2D.dat.data[:] = np.array(self.allxFSEvaluator(u1D))
        return u2D
    
    def __dampenPhiTilde__(self):
        iter = self.iter
        V1 = self.V1FS
        if iter == 0 or (self.startIteration and iter-1 == self.startIteration):
            if iter == 0:
                self.phiTarget = self.phiTilde
            else:
                self.phiTarget = fd.Function(V1)
                self.phiTarget.dat.data[:] = self.phis[0,:]

        if iter != 0:
            self.phiTilde.interpolate((1 - self.sigma) * self.phiTilde + self.sigma * self.phiTarget)
        return None
    
    @property
    def residualRatio(self):
        iter = self.iter

        if iter < 4:
            ratio = 1
        else:
            xk = self.etas[iter-1]
            xkm1 = self.etas[iter-2]
            xkm2 = self.etas[iter-3]

            yk = self.phis[iter-1]
            ykm1 = self.phis[iter-2]
            ykm2 = self.phis[iter-3]

            zk = self.ws[iter-1]
            zkm1 = self.ws[iter-2]
            zkm2 = self.ws[iter-3]

            ek = np.linalg.norm(xk - xkm1)/abs(xkm1).mean()
            ekm1 = np.linalg.norm(xkm1 - xkm2)/abs(xkm1).mean()

            pk = np.linalg.norm(yk - ykm1)/abs(ykm1).mean()
            pkm1 = np.linalg.norm(ykm1 - ykm2)/abs(ykm1).mean()

            wk = np.linalg.norm(zk - zkm1)/abs(zkm1).mean()
            wkm1 = np.linalg.norm(zkm1 - zkm2)/abs(zkm1).mean()

            Rk    = ek**2 + pk**2 + wk**2
            Rkm1  = ekm1**2 + pkm1**2 + wkm1**2
            ratio = Rk / Rkm1
        return ratio

    @property
    def dampedDT(self):
        iter = self.iter

        if iter < self.startIteration + 4:
            dampedDT = fd.Constant(self.dt)
        else:
            residual = self.residualRatio
            prevDT = self.prevDT
            
            if residual > 1.1:
                dampedDT = fd.Constant(float(prevDT) * 0.7)
            elif residual < 0.98:
                # dampedDT = min(float(prevDT)* 1.0002, self.dt)
                dampedDT = min(float(prevDT)* 1.02, self.dt*2)
                dampedDT = fd.Constant(dampedDT)
            else:
                dampedDT = prevDT
        return dampedDT
    
    def __relaxEtaAndPhi__(self, omega_eta, omega_phi):
        self.newEta.assign((1 - omega_eta) * self.eta + omega_eta * self.newEta)
        self.phiTilde.assign((1 - omega_phi) * self.phiTilde_prev + omega_phi * self.phiTilde)
        return None
    
    def __defSigma__(self):
        # For dampening phi tilde
        L_damp = fd.Constant(2.0) 
        VSigma = self.V1FS
        x = fd.SpatialCoordinate(self.fsMesh)[0]
        x_min, x_max = fd.Constant(self.xlim[0]), fd.Constant(self.xlim[1])
        L_damp = L_damp  # width of damping region at each end
        xL0 = x_min + L_damp
        xR0 = x_max - L_damp
        sigma_left = fd.cos(fd.Constant(0.5) * fd.pi * (x - x_min) / L_damp)**2
        sigma_right = fd.cos(fd.Constant(0.5) * fd.pi * (x_max - x) / L_damp)**2
        sigma_expr = fd.conditional(
            x < xL0, sigma_left,
            fd.conditional(
                x > xR0, sigma_right,
                0.0
            )
        )
        self.sigma = fd.Function(VSigma, name="sigma")
        self.sigma.interpolate(sigma_expr)
        return None

    def __buildFSSolver__(self):
        # Init FD objects for FS
        V_eta = self.V1FS
        V_phi = self.V1FS
        V_fs = V_eta*V_phi
        self.fs_n1 = fd.Function(V_fs)
        eta_n1, phi_n1 = fd.split(self.fs_n1)
        v_eta, v_phi = fd.TestFunctions(V_fs)
        self.eta_n = fd.Function(V_eta)
        self.phi_n = fd.Function(V_phi)
        g = fd.Constant(self.g)
        self.w_n = fd.Function(V_eta)
        self.wn = fd.Function(V_eta)
        One = fd.Constant(1)
        point5 = fd.Constant(0.5)
        xmin_fd, xmax_fd = fd.Constant(self.xlim[0]), fd.Constant(self.xlim[1])
        xd_in = fd.Constant(xmin_fd + 4.02112  * np.pi * self.FR**2)
        xd_out = fd.Constant(xmax_fd - 2.5 * np.pi * self.FR**2)
        x = fd.SpatialCoordinate(self.fsMesh)[0]
        A = fd.Constant(3)
        
        # Dampen eta towards the "normal" height of the domain at the edges
        eta_damp_in = A*fd.conditional(x < xd_in, ((x - xd_in) / (xmin_fd  - xd_in))**2, 0)*eta_n1
        eta_damp_out = A*fd.conditional(x > xd_out, ((x - xd_out) / (xmax_fd - xd_out))**2, 0)*eta_n1

        a_eta = fd.inner((eta_n1 - self.eta_n), v_eta)*fd.dx \
        + fd.inner(eta_damp_in + eta_damp_out, v_eta)*fd.dx

        L_eta = fd.dot(eta_n1.dx(0), phi_n1.dx(0)) \
                - self.w_n*(One + fd.dot(eta_n1.dx(0), eta_n1.dx(0)))

        F_eta = a_eta + self.dt_fd*fd.inner(L_eta, v_eta)*fd.dx

        # phi-equation
        a_phi = fd.inner((phi_n1 - self.phi_n), v_phi)*fd.dx

        L_phi = g*eta_n1 + point5*(
            fd.dot(phi_n1.dx(0), phi_n1.dx(0))
            - (self.w_n**2)*(One + fd.dot(eta_n1.dx(0), eta_n1.dx(0)))
        )

        F_phi = a_phi + self.dt_fd*fd.inner(L_phi, v_phi)*fd.dx

        self.F = F_eta + F_phi
        self.eta_bc_const = fd.Constant(0.0)
        self.DBC = [fd.DirichletBC(V_fs.sub(0), self.eta_bc_const, 1)]

        J = fd.derivative(self.F, self.fs_n1)

        self.problem = fd.NonlinearVariationalProblem(self.F, self.fs_n1,
                                                    bcs=self.DBC, J=J)

        self.FSsolver = fd.NonlinearVariationalSolver(
            self.problem,
            solver_parameters={
                "snes_max_it": self.maxItWeak1d,
                "snes_rtol":   self.tolWeak1d,
            },
        )

        self.__defSigma__()
        self.u_pot = fd.Function(self.W)
        return None

    def __weak1dFsEq__(self):
        '''
        Solves the weak form backward Euler forumulation of the phi and eta at the free surface.
        The equations are derived in the report.
        '''
        # Define previous time step functions
        self.eta_n.dat.data[:] = self.eta.dat.data_ro[:] - self.ylim[1] # Shift eta such that the eta=0 -> y = 0
        self.phi_n.dat.data[:] = self.phiTilde.dat.data_ro[:]

        # Initial guess for new time step
        self.fs_n1.sub(0).assign(self.eta_n)   # eta^{n+1} initial guess
        self.fs_n1.sub(1).assign(self.phi_n)   # phi^{n+1} initial guess

        #### Jittery dt scheme
        jitter = 0.01 * (-1)**(self.iter//2) * (2**(self.iter%6)%5)/2
        self.dt_fd.assign(self.dt * (1 + jitter))

       
        # Retrieve w_n from the pure potential phi (Avoids numerical errors in BC-correction)
        self.u_pot.project(fd.grad(self.phi))
        self.w_n.dat.data[:] = np.array(self.FSEvaluator(self.u_pot))[:,1]
        self.wn.assign(self.w_n) # For plot export

        try:
            self.FSsolver.solve()
        except:
            raise BrokenPipeError("FS equations diverged")
        

        self.phiTilde_prev.assign(self.phiTilde)
        # Extract new eta and phiTilde
        eta_sub, phi_sub = self.fs_n1.sub(0), self.fs_n1.sub(1)

        self.newEta.assign(eta_sub)
        self.phiTilde.assign(phi_sub)

        self.upperLeftValue.assign(self.upperLeftFSEvaluator(self.phiTilde)[0])
        self.phiTilde.assign(self.phiTilde - self.upperLeftValue)

        # Dampen phiTilde
        self.__dampenPhiTilde__()
        self.newEta.dat.data[:] += self.ylim[1] # Shift eta back to original position


        # ---- Relax eta and phi_tilde ----
        self.__relaxEtaAndPhi__(omega_eta=0.5, omega_phi=0.5)
        

        self.residuals = fd.norm(self.newEta - self.eta, norm_type='l2')/(1+jitter)

        self.newEta2d = None
        self.phiTilde2d = self.__lift_1d_to_2d__(self.phiTilde, self.phiTilde2d) 
        return None
    
    def __prepxy__(self, eta):
        order = np.argsort(self.coordsFS[:, 0], kind="mergesort")
        x = self.coordsFS[order,0]
        y = eta.dat.data_ro[order]
        return x, y

    def __interp1dToV__(self, eta, coords2d):
        xs, ys = self.__prepxy__(eta)
        xi = coords2d[:,0]
        return np.interp(xi, xs, ys, left=ys[0], right=ys[-1])
    
    def __gatherPointsAndDefineEvaluators__(self):
        # Gather position of points
        self.allPoints = (fd.Function(self.W).interpolate(self.mesh.coordinates).dat.data)
        self.xFS = (fd.Function(self.W1FS).interpolate(self.fsMesh.coordinates).dat.data)

        # Define evaluators for full mesh
        self.TEevaluator = fd.PointEvaluator(self.mesh, self.pointAtTE).evaluate
        self.FSEvaluator = fd.PointEvaluator(self.mesh, self.coordsFS).evaluate
        self.upperLeftEvaluator = fd.PointEvaluator(self.mesh, [self.xlim[0],self.ylim[1]], missing_points_behaviour="warn").evaluate


        # Define evaluators for fs mesh
        self.allxFSEvaluator = fd.PointEvaluator(self.fsMesh, self.allPoints[:,0]).evaluate
        self.upperLeftFSEvaluator = fd.PointEvaluator(self.fsMesh, self.xlim[0]).evaluate
        self.FSxEvaluator = fd.PointEvaluator(self.fsMesh, self.xFS).evaluate
        return None
    
    def __shiftSurface__(self):
        coords = self.mesh.coordinates.dat.data
        M = self.yInterface

        # mask points above the airfoil surface
        coordMask = coords[:, 1] >= M
        x_2d = coords[coordMask, 0]

        # Define the evaluator
        x_2devaluator = fd.PointEvaluator(self.fsMesh, x_2d).evaluate

        # FS-variables evaluated at mesh x-coordinates and keeping same order
        eta = x_2devaluator(self.eta)
        newEta = x_2devaluator(self.newEta)

        coords[coordMask, 1] = M + (newEta - M) / (eta - M) * (coords[coordMask, 1] - M)
        self.mesh.coordinates.dat.data[:] = coords
        return None
    
    def __shiftFSmesh__(self):
        fSIndecies = self.W1.boundary_nodes(4)
        self.coordsFS = (fd.Function(self.W1).interpolate(self.mesh.coordinates).dat.data)[fSIndecies,:]
        # Ensure nodes match the x-coordinates of free surface variables
        self.fsMesh.coordinates.dat.data[:] = self.coordsFS[:,0]
        return None
    
    def __updateMeshData__(self):
        # Update mesh
        self.__shiftSurface__()
        # self.__shiftSurface2DEta__()

        # Find points at free surface
        self.__shiftFSmesh__()
        
        #self.__gatherPointsAndDefineEvaluators__()
        # Update eta
        self.eta.assign(self.newEta)
        #self.eta2d.assign(self.newEta2d)
        return None
    
    def __checkStatus__(self, start_time, iteration_time):
        i = self.iter
        if (self.residuals < self.tolFreeSurface) and (i > self.minItFreeSurface):
            print(
                f"""
            {"\n" + "="*50}
             Fs converged
             residuals norm {np.linalg.norm(self.residuals)} after {i} iterations
             Total solve time: {time() - start_time}
                """
            )
            return True
        # If divergence kriteria is met print relevant information
        elif self.residuals > 10000:
            print(f"""
             Fs diverged
             residuals norm {np.linalg.norm(self.residuals)} after {i} iterations
             Total solve time: {time() - start_time}
            {"-"*50 + "\n"}
            """)
            return True
        # If the maximum amout of iterations is done print relevant information
        elif i >= self.maxItFreeSurface - 1:
            print(f"""
             Fs did not converge
             residuals norm {np.linalg.norm(self.residuals)} after {i} iterations
             Total solve time: {time() - start_time}
            {"-"*50 + "\n"}
            """)
            return True
        # If none of the above, print relevant information about solver status
        else:
            block = (
f"""\t iteration: {i+1}
\t residual norm {self.residuals}
\t iteration time: {time() - iteration_time}
{"-"*50 + "\n"}""")
            global deleteLines
            if not deleteLines:
                deleteLines = True

            print(block)
            return False 

    def solve(self):
        # Start time
        start_time = time()

        # Stop kutta condition from writing
        self.writeKutta = False

        # Saving output path
        outfileFS = self.__saveOutputPath__()
        
        # Initialize FS solver by applying kutta condition to a standard poisson solve
        if not self.startIteration:
            self.__doInitialKuttaSolve__()

        # Initialize phi tilde and eta
        self.__initPhiTilde__()
        self.__initEta__()

        self.__buildFSSolver__()
        self.__buildKuttaSolver__()

        print("Initialization done \n" + "-"*50 + "\n")
        # Start main loop
        for iteration in range(self.startIteration, self.maxItFreeSurface):
            self.iter = iteration
            # Note time for start of iteration
            iteration_time = time()

            # Calculate free surface
            if not (self.startIteration == iteration and self.startIteration):
                self.__weak1dFsEq__()
            
            self.__updateMeshData__()

            # Apply kutta condition to a poisson solve on the new mesh
            self.__doKuttaSolve__()

            # Save result
            self.__save_results__()
            if (iteration % self.outputIntervalFS) == 0:
                pressure = self.__getPressureCoefficients__()
                pressure.rename("Pressure")
                outfileFS.write(self.u, pressure)

            # Check solver status
            if self.__checkStatus__(start_time, iteration_time):
                break
            import psutil, os
            if self.iter % 100 == 0:
                p = psutil.Process(os.getpid())
                print(f"[mem] iter {self.iter}: {p.memory_info().rss/1e6:.1f} MB")
        return None


if __name__ == "__main__":
    solver = FSSolver(hypParams, meshSettings, solverSettings, outputSettings)
    solver.solve()







