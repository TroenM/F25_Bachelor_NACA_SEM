import firedrake as fd
import numpy as np
from time import time
import shutil


# Fetching own packages
import os
currdir = os.getcwd()
os.chdir("../PoissonSolver/PoissonCLS")
from poisson_solver import PoissonSolver
os.chdir("../../Meshing")
from mesh_library import*
os.chdir(currdir)



def potential_flow_solver(mesh : meshio.Mesh, P: int = 1, write : bool = True, **kwargs) -> None:
    """
    Computes potential flow and correction using kutta-condition

    Param:
    ----
    mesh: meshio.Mesh
        - The 
    """
    if write:
        if os.path.exists("./output"):
            shutil.rmtree("./output")

        try:
            os.remove("./output.pvd")
        except:
            pass


    naca_lines = mesh.cells_dict["line"][np.where(np.concatenate(mesh.cell_data["gmsh:physical"]) == kwargs.get("naca", 5))[0]]
    naca_points = np.unique(naca_lines)
    center_of_airfoil = np.array([0.5,0])
    p1 = mesh.points[np.min(naca_points)][:2]
    p2 = mesh.points[np.max(naca_points)][:2]
    p1 = (p1-center_of_airfoil)*1.2 + center_of_airfoil
    p2 = (p2-center_of_airfoil)*1.2 + center_of_airfoil

    p3 = ((p1+p2)/2)[:2]
    p12vec = (p2 - p1)
    p3_new = p3-center_of_airfoil
    fd_mesh = meshio_to_fd(mesh)
    model = PoissonSolver(fd_mesh, P)
    model.impose_NBC(kwargs.get("V_inf", fd.Constant(1.0)), kwargs.get("inlet", 1))
    model.impose_NBC(kwargs.get("V_inf", fd.Constant(1.0)), kwargs.get("outlet", 2))
    model.solve()
    u0 = model.u_sol
    velocity = fd.Function(model.W, name="initial velocity")
    velocity.project(fd.grad(u0))
    if write:
        outfile = fd.VTKFile("output.pvd")
        outfile.write(velocity)

    converged = False
    Gamma_old = 0
    vortex_sum = fd.Function(model.W, name="vortex")
    velocityBC_sum = fd.Function(model.W, name="Boundary correction")
    it = 0
    while not converged:
        t1 = time()
        print(f"Iteration {it}")
        it += 1

        v1 = velocity.at(p1)
        print(f"\t{v1}")
        v2 = velocity.at(p2)
        v3 = velocity.at(p3)
        print(f"\t{v2}")
        tev = (v1 + v2)/2
        print(f"\t old trailing edge velocity is {tev}")
        print(f"\t old trailing edge velocity at p3 is {v3}")
        # Trust me bro function
        Gamma = (p12vec[0]*tev[0] - p12vec[1]*tev[1])*2*np.pi*(p3_new[0]**2 + p3_new[1]**2)/(p12vec[1]*p3_new[0] - p12vec[0]*p3_new[1])/8
        # Check for convergence or divergence
        if (np.abs(Gamma - Gamma_old) < 10**(-3) or np.abs(Gamma - Gamma_old) > 10**(4)) and it != 1:
            converged = True
        # Creating vortex function
        vortex = fd.Function(model.W)
        # Recentering coordinates around center of airfoil
        x_new = model.x-center_of_airfoil[0]
        y_new = model.y-center_of_airfoil[1]
        vortex.project(fd.as_vector([-Gamma/2*np.pi*(y_new)/(x_new**2 + y_new**2), Gamma/2*np.pi*(x_new)/(x_new**2 + y_new**2)]))
        vortex_sum += vortex
        velocity += vortex
        # Calculating new velocity at trailing edge, to see wether the vortex alone helped
        tev_new = velocity.at(p3)
        print(f"\t vortex: {vortex.at(p3)}\n")
        print(f"\t v3 + vortex {v3+ vortex.at(p3)}")
        print(f"\t dot product {np.dot(p12vec, v3 + vortex.at(p3))}")
        
        # Creating the function 
        model = PoissonSolver(fd_mesh, P)
        model.impose_NBC( -vortex, kwargs.get("inlet", 1))
        model.impose_NBC( -vortex, kwargs.get("outlet", 2))
        model.impose_NBC( -vortex, kwargs.get("bed", 3))
        model.impose_NBC( -vortex, kwargs.get("fs", 4))
        model.impose_NBC( -vortex, kwargs.get("naca", 5))
        model.solve()
        u0 = model.u_sol
        velocityBC = fd.Function(model.W)
        velocityBC.project(fd.grad(u0))
        velocityBC_sum += velocityBC

        velocity += velocityBC
        


        if write:
            outfile.write(velocity)

        print(f"\t Itteration time: {time()-t1}")
        print(f"\t dGamma: {Gamma - Gamma_old}")
        print(f"\t Gamma: {Gamma}")
        print(f"\t new trailing edge velocity is {tev_new}")

        Gamma_old = Gamma
        

        



    if write:
        outfile.write(velocityBC_sum)
        outfile.write(vortex_sum)
    return None


            