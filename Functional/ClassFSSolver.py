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
import copy

import os
if not os.getcwd().endswith("Functional"):
    raise InterruptedError("""ClassFSSolver.py must be run from Functional folder, 
                           because Magnus is too laze to fix relative import/export paths""")

"""
IMPORTANT: The boundaries should be indexed as follows:
1: Inflow
2: Outflow
3: Bed
4: Free Surface
5: Airfoil
"""

hypParams = {
    "P": 3, # Polynomial degree
    "V_inf": fd.as_vector((10.0, 0.0)), # Free stream velocity
    "rho": 1.225 # Density of air [kg/m^3]
}

meshSettings = {
    "airfoilNumber": "0012", # NACA airfoil number
    "alpha_deg": 10, # Angle of attack in degrees
    "centerOfAirfoil": (0.5, 0), # Center of airfoil (x,y)
    "circle": True, # Whether to use a circular or elliptical vortex
    "xlim": (-8, 27), # x-limits of the domain
    "ylim": (-4, 1), # y-limits of the domain
    "nIn": 20, # Number of external nodes on inlet boundary 
    "nOut": 20, # Number of external nodes on outlet boundary
    "nBed": 150, # Number of external nodes on bed boundary
    "nFS": 150, # Number of external nodes on free surface boundary
    "nAirfoil": 200 # Number of external nodes on airfoil boundary
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
    "outputPath": "./TestResults/",
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
        self.centerOfVortex = meshSettings.get("centerOfVortex", self.centerOfAirfoil) # Default to centerOfAirfoil if not provided
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

        fSIndecies = self.V.boundary_nodes(4)
        self.coordsFS = (fd.Function(self.W).interpolate(self.mesh.coordinates).dat.data)[fSIndecies,:]

        # Output parameters
        self.outputPath = outputSettings["outputPath"]
        self.writeKutta = outputSettings["writeKutta"]
        self.writeFreeSurface = outputSettings["writeFreeSurface"]
        self.outputIntervalKutta = outputSettings["outputIntervalKutta"]
        self.outputIntervalFS = outputSettings["outputIntervalFS"]

        print("Initialized FSSolver with:\n" + f"P={self.P}\n" + 
              f"alpha={np.round(np.rad2deg(self.alpha), 2)} deg\n" + 
              f"V_inf={self.V_inf}\n" + 
              f"Degrees of freedom: {self.V.dof_count}")
        
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
            self.centerOfVortex -= np.array([vPerp[1], -vPerp[0]])/4

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
        
        fd.solve(a == L, phi, bcs=DBCs)#, solver_parameters={"pc_type": "hypre", "pc_hypre_type": "boomeramg", "ksp_rtol": 1e-10})
        
        if len(DBCs) == 0: # Normalize phi if there are no Dirichlet BCs
            phi -= np.min(phi.dat.data)
        
        u = fd.Function(self.W).interpolate(fd.grad(phi))
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
        velocityAtTE = self.u.at(self.pointAtTE) # Requires that self.u is computed before calling this function
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
        NBCs = [(i, -vortex) for i in range(1,6)] # Set NBC = [fd.as_vector(V_inf, 0)]*2 for far-field
        phiBC, uBC = self.__poissonSolver__(NBC = NBCs)

        return phiBC, uBC

    def __checkKuttaConvergence__(self, it):
        """
        Checks whether the Kutta condition has converged
        """
        velocityAtTE = self.__normaliseVector__(self.u.at(self.pointAtTE))
        dotProductTE = np.dot(velocityAtTE, self.vPerp)
        GammaDiff = np.inf if len(self.Gammas) < 2 else abs(self.Gammas[-1] - self.Gammas[-2])
        if abs(dotProductTE) < self.tolKutta:
            print(f"Kutta condition applied in {it+1} iterations")
            print(f"Dot product at TE: {dotProductTE}")
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
    
    def getLiftCoefficient(self):
        # Compute lift based on circulation given the formula in the Kutta Jacowski theorem
        Gamma = np.sum(np.array(self.Gammas))
        V_inf = np.linalg.norm(np.array(self.V_inf, dtype=float))
        lift = -Gamma * V_inf * self.rho
        lift_coeff = lift / (1/2 * self.rho * V_inf**2)
        return lift_coeff

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
                print("-"*50 + "\n")
                break

        return None
    
    #=================================================================#
    #======================== Free Surface Update ====================#
    #=================================================================#
    def __saveOutputPath__(self) -> None:
        if self.writeFreeSurface:
            if os.path.exists(self.outputPath + "FSIterations"):
                shutil.rmtree(self.outputPath + "FSIterations")

            try:
                os.remove(self.outputPath + "FSIterations.pvd")
            except:
                pass

            outfileFS = fd.VTKFile(self.outputPath + "FSIterations.pvd")
            self.u.rename("Velocity")
            return outfileFS
    
    def __doKuttaSolve__(self) -> None:
        try:
            self.phi, self.u = self.__poissonSolver__(NBC=[(i, self.V_inf) for i in [1,2]], DBC=[(4, self.phiTilde)])
        except:
            self.phi, self.u = self.__poissonSolver__(NBC = [(i, self.V_inf) for i in [1,2]])
        self.__applyKuttaCondition__()
        return None

    def __initPhiTilde__(self) -> None:
        self.phiTilde = self.phi.at(self.coordsFS)
        return None
    
    def __initEta__(self):
        self.eta = self.coordsFS[:,1]
        return None

    def __weak1dFsEq__(self):


        self.phiTilde = asd
        self.newEta = asd
        self.residuals = asd
        return None
    
    def __shiftSurface__(self):
        from firedrake.__future__ import interpolate
        mesh = self.mesh
        x, y = fd.SpatialCoordinate(mesh)
        V1 = fd.FunctionSpace(mesh, "CG", 1)
        W1 = fd.VectorFunctionSpace(mesh, "CG", 1)
        V = self.V
        W = self.W

        # Define maximal y value of coordinates on airfoil (M)
        coords = mesh.coordinates.dat.data
        naca_idx = V1.boundary_nodes(5)
        M = fd.Constant(np.max(coords[naca_idx][:,1])) 
        
        # scaling function
        s = interpolate(self.newEta/self.eta, V)

        # Shift only coords above M
        y_new = fd.conditional(fd.ge(y, M), M + s*(y-M), y)

        # Set new coordinates of mesh
        X_new = fd.project(fd.as_vector([x, y_new]), W1)
        mesh.coordinates.assign(X_new)

        self.mesh.coordinates.dat.data[:] = mesh.coordinates.dat.data
        return None

    def __updateMeshData__(self):
        # Update mesh
        self.__shiftSurface__()

        # Change the firedrake function spaces to match the new mesh
        self.V = fd.FunctionSpace(self.mesh, "CG", self.P)
        self.W = fd.VectorFunctionSpace(self.mesh, "CG", self.P)

        # Find points at free surface
        fSIndecies = self.V.boundary_nodes(4)
        self.coordsFS = (fd.Function(self.W).interpolate(self.mesh.coordinates).dat.data)[fSIndecies,:]

        # Update eta
        self.eta = self.newEta
        return None
    
    def __checkStatus__(self, i : int, start_time, iteration_time):
        if self.residuals < self.kwargs.get("fs_rtol", 1e-5):
            print("\n" + "="*50)
            print(" Fs converged")
            print(f" residuals norm {np.linalg.norm(self.residuals)} after {i} iterations")
            print(f" Total solve time: {time() - start_time}")
            return True
        # If divergence kriteria is met print relevant information
        elif self.residuals > 10000:
            print(" Fs diverged")
            print(f" residuals norm {np.linalg.norm(self.residuals)} after {i} iterations")
            print(f" Total solve time: {time() - start_time}")
            print("-"*50 + "\n")
            return True
        # If the maximum amout of iterations is done print relevant information
        elif iter >= self.kwargs.get("max_iter_fs", 10) - 1:
            print(" Fs did not converge")
            print(f" residuals norm {np.linalg.norm(self.residuals)} after {i} iterations")
            print(f" Total solve time: {time() - start_time}")
            print("-"*50 + "\n")
            return True
        # If none of the above, print relevant information about solver status
        else:
            print(f"\t iteration: {i+1}")
            print(f"\t residual norm {self.residuals}")
            print(f"\t iteration time: {time() - iteration_time}\n")
            print("-"*50 + "\n")
            return False 
    
    def solve(self):
        # Start time
        start_time = time()

        # Stop kutta condition from writing
        self.writeKutta = False

        # Saving output path
        outfileFS = self.__saveOutputPath__
        
        # Initialize FS solver by applying kutta condition to a standard poisson solve
        self.__doKuttaSolve__()

        # Initialize phi tilde and eta
        self.__initPhiTilde__()
        self.__initEta__()

        print("Initialization done")
        # Start main loop
        for iteration in range(self.maxItFreeSurface):
            # Note time for start of iteration
            iteration_time = time()

            # Calculate free surface
            self.__weak1dFsEq__()
            
            # Update mesh data
            self.__updateMeshData__()

            # Apply kutta condition to a poisson solve on the new mesh
            self.__doKuttaSolve__()

            # Save result
            outfileFS.write(self.u)

            # Check solver status
            if self.__checkStatus__(iteration, start_time, iteration_time):
                break
        return None

    #=================================================================#
    #====================== Plotting Tools ===========================#
    #=================================================================#

    def plotVelocityField(self, xlim = None, ylim = None):
        fig, ax = plt.subplots()
        
        # Plot domain boundaries
        for i in range(1, 5):
            boundaryNodes = self.V.boundary_nodes(i)
            coords = (fd.Function(self.W).interpolate(self.mesh.coordinates).dat.data)[boundaryNodes,:]
            ax.plot(coords[:,0], coords[:,1], 'k-')
        
        # plot airfoil
        airfoilCoords = naca_4digit(self.airfoilNumber, self.nAirfoil, np.rad2deg(self.alpha), self.centerOfAirfoil)
        ax.plot(airfoilCoords[:,0], airfoilCoords[:,1], 'k-')

        # Plot quiver vPerp and velocity at TE
        ax.quiver(self.pointAtTE[0], self.pointAtTE[1], self.vPerp[0], self.vPerp[1], color='r', scale=10, label='vPerp at TE')
        velocityAtTE = self.u.at(self.pointAtTE)
        ax.quiver(self.pointAtTE[0], self.pointAtTE[1], velocityAtTE[0], velocityAtTE[1], color='b', scale=10, label='Velocity at TE')

        # p1 = airfoilCoords[1]
        # pn = airfoilCoords[-1]
        # v1 = (self.TE - p1)
        # vn = (self.TE - pn)
        # ax.quiver(p1[0], p1[1], v1[0], v1[1], color='g', scale=1e-2, label='v1 at TE')
        # ax.quiver(pn[0], pn[1], vn[0], vn[1], color='m', scale=1e-2, label='vn at TE')

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        
        ax.set_aspect('equal')


if __name__ == "__main__":
    solver = FSSolver(hypParams, meshSettings, solverSettings, outputSettings)
    #changing mesh (Hopefully)
    solver.eta = fd.Constant(4)
    x, y = fd.SpatialCoordinate(solver.mesh)
    solver.newEta = 1*fd.sin(x)*x/7 + fd.Constant(8)
    solver.__updateMeshData__()


    phi, u = solver.__poissonSolver__(NBC = [(i, solver.V_inf) for i in range(1, 3)])
    solver.u = u
    solver.__applyKuttaCondition__()
    
    # solver.solve()



    #solver.plotVelocityField(xlim = (-2, 2), ylim = (-1, 1))
    #plt.show()







