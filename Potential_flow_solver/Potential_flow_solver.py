import firedrake as fd
import numpy as np


# Fetching own packages
import os
currdir = os.getcwd()
os.chdir("../PoissonSolver/PoissonCLS")
from poisson_solver import PoissonSolver
os.chdir("../../Meshing")
from mesh_library import*
os.chdir(currdir)


def potential_flow_solver(mesh : meshio.Mesh, P: int = 1, **kwargs) -> None:
    """
    Computes potential flow and correction using kutta-condition

    Param:
    ----
    mesh: meshio.Mesh
        - The 
    """
    naca_lines = mesh.cells_dict["line"][np.where(np.concatenate(mesh.cell_data["gmsh:physical"]) == kwargs.get("naca", 5))[0]]
    naca_points = np.unique(naca_lines)
    p1 = mesh.points[np.min(naca_points)]
    p2 = mesh.points[np.max(naca_points)]
    p3 = (p1+p2)/2
    fd_mesh = meshio_to_fd(mesh)
    model = PoissonSolver(fd_mesh, P)
    model.impose_NBC(kwargs.get("V_inf", fd.Constant(1.0)), kwargs.get("inlet", 1))
    model.impose_NBC(kwargs.get("V_inf", fd.Constant(1.0)), kwargs.get("outlet", 2))
    model.solve()
    u0 = model.u_sol
    velocity = fd.Function(model.W, name="initial velocity")
    velocity.project(fd.grad(u0))

    outfile = fd.VTKFile("output.pvd")
    outfile.write(velocity)

    converged = False
    Gamma_old = 100
    vortex_sum = fd.Function(model.W, name="vortex")
    velocityBC_sum = fd.Function(model.W, name="Boundary correction")
    while not converged:
        v1 = velocity.get([p1])
        v2 = velocity.get([p2])
        du = (v1 + v2)[1]
        Gamma = du*2*np.pi*(p3@p3.T/p3[1])
        if du < 10**(-3) and np.abs(Gamma - Gamma_old) < 10**(-3):
            converged = True
        vortex = fd.Function(model.W)
        vortex.project(fd.as_vector([-Gamma/2*np.pi*(model.x)/(model.x**2 + model.y**2), -Gamma/2*np.pi*(model.x)/(model.x**2 + model.y**2)]))
        vortex_sum += vortex
        velocity += vortex

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

        outfile.write(velocity)
    
    outfile.write(velocityBC_sum)
    outfile.write(vortex_sum)
    return None


            