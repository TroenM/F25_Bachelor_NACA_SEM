import firedrake as fd
import numpy as np
from time import time
import shutil


# Fetching own packages
import os
currdir = os.getcwd()
os.chdir("../../PoissonSolver/PoissonCLS")
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


    # Identifying the trailing edge
    naca_lines = mesh.cells_dict["line"][np.where(np.concatenate(mesh.cell_data["gmsh:physical"]) == kwargs.get("naca", 5))[0]]
    naca_points = np.unique(naca_lines)
    p1 = mesh.points[np.min(naca_points)][:2]
    p2 = mesh.points[np.max(naca_points)][:2]
    p_leadingedge = mesh.points[np.min(naca_points) + (np.max(naca_points) - np.min(naca_points))//2][:2]
    p_te = ((p1+p2)/2)[:2] # Trailing edge point
    center_of_airfoil = np.array([2,0.5])#p_leadingedge + (p_te-p_leadingedge)*0.8
    p1 = (p1-center_of_airfoil)*1.01 + center_of_airfoil
    p2 = (p2-center_of_airfoil)*1.01 + center_of_airfoil
    p_te = ((p1+p2)/2)[:2]
    v12 = (p2 - p1)
    p_te_new = p_te-center_of_airfoil

    # Initializing Laplaze solver
    fd_mesh = meshio_to_fd(mesh)
    model = PoissonSolver(fd_mesh, P=P)
    V_inf = kwargs.get("V_inf", 1.0)
    model.impose_NBC(fd.Constant(-V_inf), kwargs.get("inlet", 1))
    model.impose_NBC(fd.Constant(V_inf), kwargs.get("outlet", 2))
    model.solve(solver_params={"ksp_type": "preonly", "pc_type": "lu"})

    u0 = model.u_sol
    u0 -= model.u_sol.dat.data.min()
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

        v_te = velocity.at(p_te)
        print(f"\t old trailing edge velocity is {v_te}")
        
        # Trust me bro function
        Gamma = - (v12[0]*v_te[0] + v12[1]*v_te[1]) * 2*np.pi*(p_te_new[0]**2 + p_te_new[1]**2) / (v12[1]*p_te_new[0] - v12[0]*p_te_new[1])/7
        # Check for convergence or divergence
        if (np.abs(Gamma - Gamma_old) < 10**(-7) or np.abs(Gamma - Gamma_old) > 10**(4)) and it != 1:
            converged = True
            
        # Creating vortex function
        vortex = fd.Function(model.W)
        # Recentering coordinates around center of airfoil
        x_new = (model.x-center_of_airfoil[0])
        y_new = (model.y-center_of_airfoil[1])
        vortex.project(fd.as_vector([-Gamma/(2*np.pi)*(y_new)/(x_new**2 + y_new**2), Gamma/(2*np.pi)*(x_new)/(x_new**2 + y_new**2)]))
        vortex_sum += vortex
        velocity += vortex


        vortex_te_goal = np.array([-Gamma/(2*np.pi)*(p_te_new[1])/(p_te_new[0]**2+p_te_new[1]**2), Gamma/(2*np.pi)*(p_te_new[0])/(p_te_new[0]**2+p_te_new[1]**2)])
        vortex_te = vortex.at(p_te)
        # Calculating new velocity at trailing edge, to see wether the vortex alone helped
        v_te_new = velocity.at(p_te)
        print(f"\t dot product {np.dot(v12, vortex_te + v_te)}")
        
        # Creating the function 
        model = PoissonSolver(fd_mesh, P)
        model.impose_NBC( -vortex, kwargs.get("inlet", 1))
        model.impose_NBC( -vortex, kwargs.get("outlet", 2))
        model.impose_NBC( -vortex, kwargs.get("bed", 3))
        model.impose_NBC( -vortex, kwargs.get("fs", 4))
        model.impose_NBC( -vortex, kwargs.get("naca", 5))
        model.solve(solver_params={"ksp_type": "preonly", "pc_type": "lu"})
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
        print(f"\t new trailing edge velocity is {v_te_new}\n")

        Gamma_old = Gamma
        

        



    # if write:
        # outfile.write(velocityBC_sum)
        # outfile.write(vortex_sum)
    return None


######################################
# TESTING THE POTENTIAL FLOW SOLVER #
######################################

if __name__ == "__main__":
    print(os.getcwd())
    #mesh = naca_mesh("0012", alpha=20)
    #potential_flow_solver(mesh, P = 3)