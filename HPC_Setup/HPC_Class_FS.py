import os;
os.environ["OMP_NUM_THREADS"] = "1"

from GenerateMesh import getMeshSettings, gethypParams, naca_4digit
import firedrake as fd
from firedrake.pyplot import tripcolor
import numpy as np
import matplotlib.pyplot as plt
from time import time
import shutil
import sys

import os
if os.getcwd()[-21:] == 'F25_Bachelor_NACA_SEM':
    os.chdir("./HPC_Setup")
if not os.getcwd().endswith("HPC_Setup"):
    raise InterruptedError("""HPC_Class_FS.py must be run from "HPC_Setup" or root folder, 
                           because Magnus is too lazy to fix relative import/export paths""")

"""
IMPORTANT: The boundaries should be indexed as follows:
1: Inflow
2: Outflow
3: Bed
4: Free Surface
5: Airfoil
"""


hypParams = gethypParams()
meshSettings = getMeshSettings()

solverSettings = {
    "maxItKutta": 50,
    "tolKutta": 1e-10,
    "maxItFreeSurface": 20000,
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
    "outputIntervalFS": 1000, # Output interval in free surface time steps
    "writeArraysInterval": 1000
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
        self.mesh = fd.Mesh("mesh.msh")
        self.yInterface, *self.ylim = np.load("y_data.npy")
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

        self.FR = hypParams["FR"]
        self.g = (self.V_inf[0]**2+self.V_inf[1]**2)/self.FR**2

        # Output parameters
        self.outputPath = outputSettings["outputPath"]
        self.writeKutta = outputSettings["writeKutta"]
        self.writeFreeSurface = outputSettings["writeFreeSurface"]
        self.outputIntervalKutta = outputSettings["outputIntervalKutta"]
        self.outputIntervalFS = outputSettings["outputIntervalFS"]
        self.writeArraysInterval = outputSettings["writeArraysInterval"]

        self.startIteration = solverSettings.get("startIteration", 0)
        
        if not self.startIteration:
            print("Starting new solve")
            self.etas = np.zeros((self.maxItFreeSurface+1, len(self.coordsFS)), dtype=np.float64)
            self.phis = np.zeros((self.maxItFreeSurface+1, len(self.coordsFS)), dtype=np.float64)
            self.ws = np.zeros((self.maxItFreeSurface+1, len(self.coordsFS)), dtype=np.float64)
            self.coordsFS_array = np.zeros((self.maxItFreeSurface+1, len(self.coordsFS)))
            self.residual_array = np.zeros((self.maxItFreeSurface+1, 2))
        else:
            print("Continued solve at iteration: ", self.startIteration)
            self.etas = np.load("TestResults/arrays/eta.npy")
            self.phis = np.load("TestResults/arrays/phiTilde.npy")
            self.ws = np.load("TestResults/arrays/ws.npy")
            self.coordsFS_array = np.load("TestResults/arrays/coordsFS.npy")
            self.residual_array = np.load("TestResults/arrays/residuals.npy")


        
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
        alpha = fd.Constant(self.alpha)
        a = fd.Constant(self.a)
        b = fd.Constant(self.b)

        # Gather coordinates from fd mesh
        fd_x, fd_y = fd.SpatialCoordinate(self.mesh)

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
        vortex = fd.Function(self.W)
        vortex.project(fd.as_vector([u_xFinal, u_yFinal]))
        self.vortex = vortex

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

        self.etas[iter, :] = self.FSxEvaluator(self.eta)
        self.phis[iter, :] = self.FSEvaluator(self.phi)
        self.ws[iter, :] = self.wn.dat.data_ro[:]
        self.coordsFS_array[iter, :] = self.fsMesh.coordinates.dat.data_ro.copy()
        self.residual_array[iter] = np.array([self.tolFreeSurface,-1]) if iter == 0 else np.array([self.residuals.copy(), self.dt*(iter)])

        if iter%self.writeArraysInterval == 0:
            np.save(self.outputPath + "arrays/eta.npy", self.etas)
            np.save(self.outputPath + "arrays/phiTilde.npy", self.phis)
            np.save(self.outputPath + "arrays/ws.npy", self.ws)
            np.save(self.outputPath + "arrays/coordsFS.npy", self.coordsFS_array)
            np.save(self.outputPath + "arrays/residuals.npy", self.residual_array)
        return None

    def __doKuttaSolve__(self) -> None:
        '''iter is tool for testing'''
        if hasattr(self, "phiTilde2d"):
            self.phi, self.u = self.__poissonSolver__(NBC=[(i, self.V_inf) for i in [1,2]], DBC=[(4, self.phiTilde2d)])
        else:
            self.phi, self.u = self.__poissonSolver__(NBC=[(i, self.V_inf) for i in [1,2]])
        self.__applyKuttaCondition__()
        return None

    def __initPhiTilde__(self) -> None:
        V1 = self.V1FS
        if not self.startIteration:
            self.phiTilde = fd.Function(V1)
            self.phiTilde_prev = fd.Function(V1)
            self.phiTilde.dat.data[:] = self.FSEvaluator(self.phi)
            self.phiTilde_prev.dat.data[:] = self.FSEvaluator(self.phi)
        else:
            self.phiTilde = fd.Function(V1)
            self.phiTilde_prev = fd.Function(V1)

            self.phiTilde.dat.data[:] = self.phis[self.startIteration,:]
            self.phiTilde2d = self.__lift_1d_to_2d__(self.phiTilde)

            self.phiTilde_prev.dat.data[:] = self.phis[self.startIteration-1,:]

        self.inletValue = fd.Constant(self.phiTilde.dat.data_ro[self.coordsFS[:,0].argmin()]) # phiTilde = constant at inflow boundary
        return None
    
    def __initEta__(self):
        V1 = self.V1FS
        if not self.startIteration:
            self.eta = fd.Function(V1).interpolate(fd.Constant(self.ylim[1]))
            self.eta2d = fd.Function(self.V).interpolate(fd.Constant(self.ylim[1]))
        else:
            self.eta = fd.Function(V1)
            self.newEta = fd.Function(V1)
            self.wn = fd.Function(V1)

            self.eta.dat.data[:] = self.etas[self.startIteration-1,:]
            self.eta2d = self.__lift_1d_to_2d__(self.eta)

            self.newEta.dat.data[:] = self.etas[self.startIteration,:]
            self.newEta2d = self.__lift_1d_to_2d__(self.newEta)

            self.residuals = fd.norm(self.newEta - self.eta, norm_type='l2')

            self.wn.dat.data[:] = self.ws[self.startIteration, :]
        return None

    def __lift_1d_to_2d__(self, u1D):
        """
        Lift a scalar 1D Firedrake Function u1D(x) to a 2D Function on self.mesh
        by defining u2D(x,y) := u1D(x).
        """
        V2 = self.V
        u2D = fd.Function(V2)

        u2D.dat.data[:] = np.array(self.allxFSEvaluator(u1D))

        return u2D
    
    def __dampenPhiTilde__(self, fsMesh):
        iter = self.iter
        V1 = self.V1FS
        if iter == 0 or (self.startIteration and iter-1 == self.startIteration):
            if iter == 0:
                self.phiTarget = self.phiTilde
            else:
                self.phiTarget = fd.Function(V1)
                self.phiTarget.dat.data[:] = self.phis[0,:]

            VSigma = V1
            x = fd.SpatialCoordinate(fsMesh)[0]
            x_min = fd.Constant(self.xlim[0])
            x_max = fd.Constant(self.xlim[1])
            L_damp = fd.Constant(2.0)   # width of damping region at each end

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
    def __relaxEtaAndPhi__(self, omega_eta, omega_phi, phi_old):
        self.newEta.assign((1 - omega_eta) * self.eta + omega_eta * self.newEta)
        self.phiTilde.assign((1 - omega_phi) * phi_old + omega_phi * self.phiTilde)
        return None
    
    def __weak1dFsEq__(self):
        '''
        Solves the weak form backward Euler forumulation of the phi and eta at the free surface.
        The equations are derived in the report.
        '''
        iter = self.iter
        fsMesh = self.fsMesh
        # Define function spaces for eta and phiTilde
        V_eta = self.V1FS
        V_phi = self.V1FS
        V_fs = V_eta*V_phi
        fs_n1 = fd.Function(V_fs)
        eta_n1, phi_n1 = fd.split(fs_n1)
        v_eta, v_phi = fd.TestFunctions(V_fs)

        # Define previous time step functions
        eta_n = fd.Function(V_eta)
        eta_n.dat.data[:] = self.eta.dat.data_ro[:] - self.ylim[1] # Shift eta such that the eta=0 -> y = 0
        phi_n = fd.Function(V_phi)
        phi_n.dat.data[:] = self.phiTilde.dat.data_ro[:]

        # Initial guess for new time step
        fs_n1.sub(0).assign(eta_n)   # eta^{n+1} initial guess
        fs_n1.sub(1).assign(phi_n)   # phi^{n+1} initial guess

        #### Static dt scheme
        jitter = 0.01 * (-1)**(iter//2) * (2**(iter%6)%5)/2
        dt = fd.Constant(self.dt * (1 + jitter))

        #### Feedback controlled dt scheme
        # dt = self.dampedDT
        # self.prevDT = fd.Constant(dt)
        # print(dt)
        # print(self.residualRatio)

        # Define additional constants and parameters
        g = fd.Constant(self.g)
        w_n = fd.Function(V_eta)
        # Retrieve w_n from the pure potential phi (Avoids numerical errors in BC-correction)
        u_pot = fd.Function(self.W)
        u_pot.project(fd.grad(self.phi))
        w_n.dat.data[:] = np.array(self.FSEvaluator(u_pot))[:,1]
        self.wn = w_n.copy() # For plot export

        One = fd.Constant(1)
        point5 = fd.Constant(0.5)

        # Define dampening parameters
        xmin, xmax = fd.Constant(self.xlim[0]), fd.Constant(self.xlim[1])
        xd_in = fd.Constant(xmin + 4.02112  * np.pi * self.FR**2) 
        xd_out = fd.Constant(xmax - 2.5 * np.pi * self.FR**2)
        x = fd.SpatialCoordinate(fsMesh)[0]
        A = fd.Constant(3)

        # Dampen eta towards the "normal" height of the domain at the edges
        eta_damp_in = A*fd.conditional(x < xd_in, ((x - xd_in) / (xmin  - xd_in))**2, 0)*eta_n1
        eta_damp_out = A*fd.conditional(x > xd_out, ((x - xd_out) / (xmax - xd_out))**2, 0)*eta_n1

        # Define variational problem
        a_eta = fd.inner((eta_n1 - eta_n), v_eta)*fd.dx + fd.inner(eta_damp_in + eta_damp_out, v_eta)*fd.dx
        L_eta = fd.dot(eta_n1.dx(0), phi_n1.dx(0)) - w_n*(One + fd.dot(eta_n1.dx(0), eta_n1.dx(0)))
        F_eta = a_eta + dt*fd.inner(L_eta, v_eta)*fd.dx

        a_phi = fd.inner((phi_n1 - phi_n), v_phi) * fd.dx
        L_phi = g*eta_n1 + point5*(fd.dot(phi_n1.dx(0), phi_n1.dx(0)) - (w_n**2)*(One + fd.dot(eta_n1.dx(0), eta_n1.dx(0))))
        F_phi = a_phi + dt*fd.inner(L_phi, v_phi)*fd.dx

        F = F_eta + F_phi

        # Boundary conditions
        DBC = []
        DBC.append(fd.DirichletBC(V_fs.sub(0), fd.Constant(0), 1)) # eta = 0 at inflow
        # DBC.append(fd.DirichletBC(V_fs.sub(1), self.inletValue, 2)) # phiTilde = constant at outflow

        # Solve variational problem
        Jacobian = fd.derivative(F, fs_n1)
        solver_parameters = {"snes_max_it": self.maxItWeak1d, "snes_rtol": self.tolWeak1d}

        try:
            fd.solve(F == 0, fs_n1, bcs=DBC, J = Jacobian, solver_parameters=solver_parameters)
        except:
            raise BrokenPipeError("FS equations diverged")
        
        phi_old = self.phiTilde
        # Extract new eta and phiTilde
        self.newEta, self.phiTilde = fs_n1.sub(0), fs_n1.sub(1)

        upperLeftValue = fd.Constant(self.upperLeftFSEvaluator(self.phiTilde)[0])
        self.phiTilde.assign(self.phiTilde - upperLeftValue)

        # Dampen phiTilde
        self.__dampenPhiTilde__(fsMesh)
        self.newEta.dat.data[:] += self.ylim[1] # Shift eta back to original position


        # ---- UNDER-RELAXATION HER ----
        self.__relaxEtaAndPhi__(omega_eta=0.3, omega_phi=0.3, phi_old=phi_old)
        

        self.residuals = fd.norm(self.newEta - self.eta, norm_type='l2')/(1+jitter)

        self.newEta2d = None
        self.phiTilde2d = self.__lift_1d_to_2d__(self.phiTilde)
        
        return None
    
    def __relax_phiTilde__(self, iter):
        """Relaxes the solution of phiTilde to avoid erratic behavior during first couple of iterations"""
        relax_start = 0.3
        theta = fd.Constant(np.min([relax_start + (1-relax_start)/500 * iter, 1]))
        self.phiTilde.project(theta*self.phiTilde + (1-theta)*self.phiTilde_prev)
        self.phiTilde_prev = self.phiTilde.copy()
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
        self.eta = self.newEta
        self.eta2d = self.newEta2d
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
            # global deleteLines
            # if not deleteLines:
                # deleteLines = True

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
            self.__doKuttaSolve__()

        # Initialize phi tilde and eta
        self.__initPhiTilde__()
        self.__initEta__()

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
        return None


if __name__ == "__main__":
    solver = FSSolver(hypParams, meshSettings, solverSettings, outputSettings)
    solver.solve()