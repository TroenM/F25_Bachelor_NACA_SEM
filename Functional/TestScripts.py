
import os
if not os.getcwd().endswith("Functional"):
    raise InterruptedError("TestScripts.py must be run from Functional folder")


option = input("""
Select an option:
1: MMS on Poisson Solver
               
Enter option: """)

if option == "1":
    from FunctionalFSSolver import PoissonSolver
    import firedrake as fd
    import numpy as np
    Ns = np.array([10, 20, 30, 50, 100]) # Number of points on each row
    ps = [1,2,3] # Polynomial orders

    errors = np.empty((len(ps), len(Ns)))
    dof = np.empty((len(ps), len(Ns)))

    for i, p in enumerate(ps):
        for j, N in enumerate(Ns):
            N = int(N/p)  # Adjust N based on polynomial order
            mesh = fd.UnitSquareMesh(N, N)
            V = fd.FunctionSpace(mesh, "CG", p)
            x, y = fd.SpatialCoordinate(mesh)
            phi_exact = fd.sin(x)*fd.sin(y)
            f = 2*fd.sin(x)*fd.sin(y)
            DBCs = [(3, phi_exact), (4, phi_exact)]
            NBCs = [(1, fd.grad(phi_exact)), (2, fd.grad(phi_exact))]
            phi_approx = PoissonSolver(mesh, V, rhs=f, DBC=DBCs, NBC=NBCs)

            error = fd.errornorm(phi_exact, phi_approx, norm_type='L2')

            errors[i,j] = error
            dof[i,j] = V.dof_count
            print(f"p = {p}, N = {N}, error = {error}")
    
    write = input("Write errors to file? (y/n): ").lower()
    if write == "y":
        np.savetxt("./TestResults/PoissonMMS/MMS_Poisson_Errors.txt", errors)
        np.savetxt("./TestResults/PoissonMMS/MMS_Poisson_DOF.txt", dof)
        print("Errors written to MMS_Poisson_Errors.txt")
    else:
        print("Errors not written to file.")