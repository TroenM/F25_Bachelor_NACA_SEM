import numpy as np
import firedrake as fd
import shutil
import os
import sys
from time import time

#### Running from F25_Bachelor_NACA_SEM ####
sys.path.append(os.getcwd())
from PoissonSolver.PoissonCLS.poisson_solver import PoissonSolver
from Meshing.mesh_library import *

os.chdir("./Potential_flow_solver/PotentialFlowSolverCLS")

class PotentialFlowSolver:
    """
    Class for solving potential flow around an airfoil with kutta condition using oval vortecies

    Params:
    ----
    airfoil : str
        - 4 digit code for NACA airfoil to be used
    P : int
        - Polynomial degree of the spectral element space
    alpha : float
        - Angle of attack of the airfoil in degrees
    V_inf : float
        - Freestream velocity

    **kwargs:
    ----
    xlim : list[float]
        - x limits of the mesh, Default: [-7, 13]
    ylim : list[float]
        - y limits of the mesh, Default: [-2, 1]

    write : bool, Default: True
        - Write output to file

    max_iter : int, Default: 20
        - Maximum number of iterations

    inlet : int, Default: 1
        - Index for inlet boundary
    outlet : int, Default: 2
        - Index for outlet boundary
    fs : int, Default: 3
        - Index for free surface boundary
    naca : int, Default: 5
        - Index for NACA airfoil boundary    

    """

    ########### Constructor ###########
    def __init__(self, airfoil : str = "0012", P : int = 1, alpha : float = 0, V_inf : float = 1.0, **kwargs):
        self.airfoil = airfoil
        self.P = P
        self.V_inf = V_inf
        self.kwargs = kwargs
        self.alpha = alpha
        self.center_of_airfoil = self.kwargs.get("center_of_airfoil", np.array([0.5,0]))

        self.write = self.kwargs.get("write", True)

        # Setting up the mesh
        self.xlim = self.kwargs.get("xlim", [-7, 13])
        self.ylim = self.kwargs.get("ylim", [-2, 1])

        self.mesh = naca_mesh(self.airfoil, self.alpha, self.xlim, self.ylim, center_of_airfoil=self.center_of_airfoil)
        self.fd_mesh = meshio_to_fd(self.mesh)

        # Handeling output files
        if self.write:
            if os.path.exists("./velocity_output"):
                shutil.rmtree("./velocity_output")
            if os.path.exists("./vortex_output"):
                shutil.rmtree("./vortex_output")
            try:
                os.remove("./velocity_output.pvd")
            except:
                pass
            try:
                os.remove("./vortex_output.pvd")
            except:
                pass
            
            self.velocity_output = fd.VTKFile("velocity_output.pvd")
            self.vortex_output = fd.VTKFile("vortex_output.pvd")


    def solve(self):
        center_of_vortex = self.kwargs.get("center_of_vortex", self.center_of_airfoil)
        # Identify trailing edge and leading edge
        p1, p2, p_te, p_leading_edge= self.get_edge_info()
        v12 = (p2 - p1)
        p_te_new = (p_te-center_of_vortex)*1.1 + center_of_vortex



        # Initializing Laplaze solver
        model = PoissonSolver(self.fd_mesh, P=self.P)
        model.impose_NBC(fd.Constant(-self.V_inf), self.kwargs.get("inlet", 1))
        model.impose_NBC(fd.Constant(self.V_inf), self.kwargs.get("outlet", 2))
        model.solve(solver_params=self.kwargs.get("solver_params", {"ksp_type": "preonly", "pc_type": "lu"}))

        # Standardizing the velocity potential to avoid overflow
        velocityPotential = model.u_sol
        velocityPotential -= model.u_sol.dat.data.min()

        # Computing the velocity field
        velocity = fd.Function(model.W, name="velocity")
        velocity.project(fd.grad(velocityPotential))
        vortex = fd.Function(model.W, name="vortex")

        if self.write:
            self.velocity_output.write(velocity)

        # Initializing main loop
        old_Gamma = 0
        vortex_sum = fd.Function(model.W, name="vortex")
        velocityBC_sum = fd.Function(model.W, name="Boundary correction")
        time_total = time()

        # Main loop
        for it, _ in enumerate(range(self.kwargs.get("max_iter", 20))):
            time_it = time()
            print(f"Starting iteration {it}")

            # Computing the vortex strength
            vte = velocity.at(p_te_new)
            Gamma = self.compute_vortex_strength(v12, vte, p_te_new) # TO BE IMPLEMENTED

            # Checking for convergence
            if np.abs(Gamma - old_Gamma) < 1e-6:
                print(f"Solver converged in {it} iterations")
                print(f"\t Total time: {time() - time_total}")
                print(f"\t dGamma: {np.abs(Gamma - old_Gamma)}")
                print(f"\n")
                break

            # Checking for divergence
            if np.abs(Gamma - old_Gamma) > 1e4:
                print(f"Solver diverged in {it} iterations")
                print(f"\t Total time: {time() - time_total}")
                print(f"\t dGamma: {np.abs(Gamma - old_Gamma)}")
                print(f"\n")
                break

            # Compute the vortex
            vortex = self.compute_vortex(Gamma/100, model, center_of_vortex, vortex) # TO BE IMPLEMENTED
            velocity += vortex
            vortex_sum += vortex

            # Computing the boundary correction
            velocityBC = self.compute_boundary_correction(vortex)
            velocity += velocityBC
            velocityBC_sum += velocityBC
            
            
            # Updating the vortex strength
            print(f"\t dGamma: {Gamma - old_Gamma}")
            old_Gamma = Gamma

            # Write to file
            if self.write:
                self.velocity_output.write(velocity)
                self.vortex_output.write(vortex)

            print(f"\t Iteration time: {time() - time_it} seconds\n")

    
    def get_edge_info(self):
        """
        Returns the coordinates of the leading edge, trailing edge and the point at the trailing edge
        """
        # fetching points on the NACA airfoil
        naca_lines = self.mesh.cells_dict["line"][np.where(
            np.concatenate(self.mesh.cell_data["gmsh:physical"]) == self.kwargs.get("naca", 5))[0]]
        naca_points = np.unique(naca_lines)

        p1 = self.mesh.points[np.min(naca_points)][:2]
        p2 = self.mesh.points[np.max(naca_points)][:2]
        p_te = ((p1+p2)/2)[:2] # Shift off boundary

        p_leading_edge = self.mesh.points[np.min(naca_points) + (np.max(naca_points) - np.min(naca_points))//2][:2]
        return p1, p2, p_te, p_leading_edge

    def compute_vortex_strength(self, v12, vte, p_te_new) -> float:
        """
        Computes the vortex strength for the given iteration
        """
        a = self.kwargs.get("a", 1)
        b = self.kwargs.get("b", int(self.airfoil[2:])/100)
        x = p_te_new[0]
        y = p_te_new[1]
        alpha = self.alpha
        Gamma = -(v12[0]*vte[0] + v12[1]*vte[1])*2*np.pi*(a**2*x**2 + b**2*y**2) / (v12[0]*(-b**2*y*np.cos(alpha) + a**2*x*np.sin(alpha)) + v12[1]*(b**2*y*np.sin(alpha) + a**2*x*np.cos(alpha)))
        return Gamma

    def compute_vortex(self, Gamma, model, center_of_vortex, vortex) -> fd.Function:
        """
        Computes the vortex field for the given vortex strength
        """
        rot_mat = np.array([
            [np.cos(self.alpha), -np.sin(self.alpha)],
            [np.sin(self.alpha), np.cos(self.alpha)]
        ])
        a = self.kwargs.get("a", 1)
        b = self.kwargs.get("b", int(self.airfoil[2:])/100)
        x_new = model.x - center_of_vortex[0]
        y_new = model.y - center_of_vortex[1]
        eliptic_vortex = np.array([-Gamma/(2*np.pi) * b**2*y_new/(a**2*x_new**2 + b**2*y_new**2) , Gamma/(2*np.pi) * a**2*x_new/(a**2*x_new**2 + b**2*y_new**2)])
        eliptic_vortex = (eliptic_vortex.T@rot_mat).T

        # Convert to Firedrake before returning
        vortex.project(fd.as_vector(eliptic_vortex))
        return vortex

    def compute_boundary_correction(self, vortex) -> fd.Function:
        """
        Computes the boundary correction for the velocity field
        """
        # Initializing the correction model
        correction_model = PoissonSolver(self.fd_mesh, self.P)

        # Imposing the Neumann boundary conditions
        correction_model.impose_NBC( -vortex, self.kwargs.get("inlet", 1))
        correction_model.impose_NBC( -vortex, self.kwargs.get("outlet", 2))
        correction_model.impose_NBC( -vortex, self.kwargs.get("bed", 3))
        correction_model.impose_NBC( -vortex, self.kwargs.get("fs", 4))
        correction_model.impose_NBC( -vortex, self.kwargs.get("naca", 5))

        # Solving the correction model
        correction_model.solve(solver_params={"ksp_type": "preonly", "pc_type": "lu"})
        velocityPotential = correction_model.u_sol
        velocityPotential -= correction_model.u_sol.dat.data.min()

        # Computing the boundary correcting velocity field
        velocityBC = fd.Function(correction_model.W)
        velocityBC.project(fd.grad(velocityPotential))

        return velocityBC



