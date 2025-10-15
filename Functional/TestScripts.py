
import os
if not os.getcwd().endswith("Functional"):
   raise InterruptedError("TestScripts.py must be run from Functional folder")

from ClassFSSolver import *
import firedrake as fd
import numpy as np

option = input("""
Select an option:
1: MMS on Poisson Solver
2: Pressure coefficient on NACA0012 airfoil (not implemented)
3: Lift coefficient plot on NACA0012 airfoil (Abbot Re=6mil)
4: Lift coefficient convergence on NACA0012 airfoil (not implemented)
               
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
    
elif option == "2":
    raise NotImplementedError("Pressure coefficient on NACA0012 airfoil not implemented yet.")

elif option == "3":
    Abbot = np.loadtxt("./TestResults/LiftCoeff/AbbotRe6mil.txt")
    alphas = Abbot[:,0]

    computed_lift = np.empty_like(Abbot)

    for it, alpha in enumerate(alphas):
        hypParams["P"] = 2
        hypParams["V_inf"] = fd.as_vector((10.0, 0.0))

        meshSettings['alpha_deg'] = alpha
        meshSettings["nIn"] = 20
        meshSettings["nOut"] = 20
        meshSettings["nFS"] = 150
        meshSettings["nBed"] = 150
        meshSettings["nAirfoil"] = 300
        meshSettings["xlim"] = (-7, 13)
        meshSettings["ylim"] = (-7, 7)
        meshSettings["circle"] = False


        outputSettings["writeKutta"] = True
        outputSettings["writeFreeSurface"] = False
        solver = FSSolver(hypParams, meshSettings, solverSettings, outputSettings)

        phi, u = solver.__poissonSolver__(NBC = [(i, solver.V_inf) for i in range(1, 3)])
        solver.u = u
        solver.__applyKuttaCondition__()
        lift_coeff = solver.getLiftCoefficient()
        computed_lift[it] = np.array([alpha, lift_coeff])
        print(f"Alpha: {alpha}, Lift Coefficient: {lift_coeff}")
        print("-"*50)

    write = input("Write lift coefficients to file? (y/n): ").lower()
    if write == "y":
        np.savetxt("./TestResults/LiftCoeff/CL" + "Circular"*solver.circle + "Elliptical"*(1 - solver.circle) +f"P{solver.P}.txt", computed_lift)
        print("Lift coefficients written to CL" + "Circular"*solver.circle + "Elliptical"*(1 - solver.circle) +f"P{solver.P}.txt")
    else:
        print("Lift coefficients not written to file.")

elif option == "4":
    raise NotImplementedError("Lift coefficient convergence on NACA0012 airfoil not implemented yet.")