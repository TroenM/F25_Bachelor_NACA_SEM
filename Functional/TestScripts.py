
import os
if not os.getcwd().endswith("Functional"):
    raise InterruptedError("TestScripts.py must be run from Functional folder")

from ClassFSSolver import FSSolver
import firedrake as fd
import numpy as np

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

option = input("""
Select an option:
1: MMS on Poisson Solver
               
Enter option: """)

if option == "1":
    Ns = np.array([10, 20, 30, 50, 100]) # Number of points on each row
    ps = [1,2,3] # Polynomial orders

    errors = np.empty((len(ps), len(Ns)))
    dof = np.empty((len(ps), len(Ns)))

    for i, p in enumerate(ps):
        for j, N in enumerate(Ns):
            solver = FSSolver(hypParams, meshSettings, solverSettings, outputSettings)
            N = int(N/p)  # Adjust N based on polynomial order
            mesh = fd.UnitSquareMesh(N, N)
            solver.mesh = mesh
            solver.V = fd.FunctionSpace(mesh, "CG", p)
            solver.W = fd.VectorFunctionSpace(mesh, "CG", p)
            x, y = fd.SpatialCoordinate(solver.mesh)
            phi_exact = fd.sin(x)*fd.sin(y)
            f = 2*fd.sin(x)*fd.sin(y)
            DBCs = [(3, phi_exact), (4, phi_exact)]
            NBCs = [(1, fd.grad(phi_exact)), (2, fd.grad(phi_exact))]
            phi_approx, u = solver.__poissonSolver__(rhs=f, DBC=DBCs, NBC=NBCs)

            error = fd.errornorm(phi_exact, phi_approx, norm_type='L2')

            errors[i,j] = error
            dof[i,j] = solver.V.dof_count
            print(f"p = {p}, N = {N}, error = {error}")
    
    write = input("Write errors to file? (y/n): ").lower()
    if write == "y":
        np.savetxt("./TestResults/PoissonMMS/MMS_Poisson_Errors.txt", errors)
        np.savetxt("./TestResults/PoissonMMS/MMS_Poisson_DOF.txt", dof)
        print("Errors written to MMS_Poisson_Errors.txt")
    else:
        print("Errors not written to file.")