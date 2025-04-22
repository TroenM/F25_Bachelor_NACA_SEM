import os
if os.getcwd().split("/")[-1] == "Fs_solver":
    os.chdir("../")
elif os.getcwd() == "/root/MyProjects":
    os.chdir("./F25_Bachelor_NACA_SEM")
elif os.getcwd() == "\home/firedrake/Projects":
    os.chdir("./Bachelor/F25_Bachelor_NACA_SEM")
 
import sys
sys.path.append(os.getcwd())
currdir = os.getcwd()

import numpy as np
import firedrake as fd
import shutil
import os
import sys
from time import time
from scipy.integrate import trapezoid

#### Running from F25_Bachelor_NACA_SEM ####
if os.getcwd().split("/")[-1] == "Projects":
    os.chdir("./Bachelor/F25_Bachelor_NACA_SEM")

print(os.getcwd())
sys.path.append(os.getcwd())

from Potential_flow_solver.PotentialFlowSolverCLS.PotentialFlowSolver import *

os.chdir("../../Fs_solver")



class FsSolver:
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
        - Maximum number of iterations on the kutta kondition
    max_iter_fs : int, Default 10
        - Maximum number of iterations on the free surface

    inlet : int, Default: 1
        - Index for inlet boundary
    outlet : int, Default: 2
        - Index for outlet boundary
    bed : int, Default 3
        - Index for the seabed
    fs : int, Default: 4
        - Index for free surface boundary
    naca : int, Default: 5
        - Index for NACA airfoil boundary    

    """

    ########### Constructor ###########
    def __init__(self, airfoil : str = "0012", P : int = 1, alpha : float = 0, **kwargs):
        self.airfoil = airfoil
        self.P = P

        try:
            self.kwargs = kwargs["kwargs"]
        except:
            self.kwargs = kwargs
        self.V_inf = self.kwargs.get("V_inf", 1.0)
        self.alpha = alpha
        self.center_of_airfoil = self.kwargs.get("center_of_airfoil", np.array([0.5,0]))
        self.Gamma = 0

        self.write = self.kwargs.get("write", True)
        self.rot_mat = np.array([
            [np.cos(self.alpha), -np.sin(self.alpha)],
            [np.sin(self.alpha), np.cos(self.alpha)]
        ])
        self.inv_rot_mat = np.array([
            [np.cos(-self.alpha), -np.sin(-self.alpha)],
            [np.sin(-self.alpha), np.cos(-self.alpha)]
        ])
        # Setting up the mesh
        self.xlim = self.kwargs.get("xlim", [-7, 13])
        self.ylim = self.kwargs.get("ylim", [-2, 1])

        self.mesh = naca_mesh(self.airfoil, self.alpha, self.xlim, self.ylim, 
                              center_of_airfoil=self.center_of_airfoil,
                              n_airfoil = self.kwargs.get("n_airfoil"),
                              n_fs = self.kwargs.get("n_fs"),
                              n_bed = self.kwargs.get("n_bed"),
                              n_inlet = self.kwargs.get("n_inlet"),
                              n_outlet = self.kwargs.get("n_outlet"))
        self.fd_mesh = meshio_to_fd(self.mesh)
        self.V = fd.FunctionSpace(self.fd_mesh, "CG", self.P)
        self.W = fd.VectorFunctionSpace(self.fd_mesh, "CG", self.P)
        fs_indecies = self.V.boundary_nodes(self.kwargs.get("fs", 4))
        self.fs_points = (fd.Function(self.W).interpolate(self.fd_mesh.coordinates).dat.data)[fs_indecies,:]
        self.fs_points = self.fs_points[self.fs_points[:,0].argsort()]
        self.fs_xs = self.fs_points[:,0]

        self.etas = []

        self.a = self.kwargs.get("a", 1)
        self.b = self.kwargs.get("b", int(self.airfoil[2:])/100)

        self.dt = self.kwargs.get("dt", 0.001)

        self.visualisationpath = "../Visualisation/FS/"

        
        # Handeling output files
        if self.write:

            if os.path.exists(self.visualisationpath + "velocity_output"):
                shutil.rmtree(self.visualisationpath + "velocity_output")

            try:
                os.remove(self.visualisationpath + "velocity_output.pvd")
            except:
                pass
            
            self.velocity_output = fd.VTKFile(self.visualisationpath + "velocity_output.pvd")


    def solve(self) -> None:
        solve_time = time()
        # Setting up kwargs for inner solver
        kwargs_for_Kutta_kondition = (self.kwargs).copy()
        kwargs_for_Kutta_kondition["write"] = False
        kwargs_for_Kutta_kondition["mesh"] = self.mesh
        kwargs_for_Kutta_kondition["fd_mesh"] = self.fd_mesh

        # Doing the initializing solve
        model = PotentialFlowSolver(self.airfoil , self.P, self.alpha, kwargs=kwargs_for_Kutta_kondition)
        model.solve()

        if self.write:
            self.velocity_output.write(model.velocity)

        # Initialize phi tilde
        self.__init_PhiTilde__(model)
        self.init_phi = self.PhiTilde.copy() # Remove after debug
        

        # Preparing for loop
        old_eta = self.fs_points[:,1]
        print("initialization done")
        for i in range(self.kwargs.get("max_iter_fs", 10)):
            iter_time = time()
            # Update free surface
            new_eta, residuals = self.__compute_eta(model)

            # Update dirichlet boundary condition
            self.__compute_phi_tilde(model)
            kwargs_for_Kutta_kondition["fs_DBC"] = self.PhiTilde

            # Update mesh
            self.__update_mesh_data(old_eta, new_eta)
            kwargs_for_Kutta_kondition["mesh"] = self.mesh
            kwargs_for_Kutta_kondition["fd_mesh"] = self.fd_mesh

            # Solve model and kutta kondition again
            model = PotentialFlowSolver(self.airfoil , self.P, self.alpha, kwargs=kwargs_for_Kutta_kondition)
            model.solve()

            self.velocity = model.velocity
            old_eta = new_eta.copy()
            if self.write:
                self.velocity_output.write(self.velocity)
            
            if self.__check_status__(residuals,i,iter_time,solve_time):
                break
        
        return None

            

    def __compute_eta(self, model : PotentialFlowSolver) -> np.ndarray:
        x = self.fs_points[:,0]
        fs_velocity = np.array(model.velocity.at(self.fs_points))[1:]
        eta = self.fs_points[:, 1]
        dt = self.dt

        # Compute the x-distance between all nodes
        dx = x[1:] - x[:-1] #dx = x_{i} - x_{i-1}

        # First order x-stencil for eta_x
        eta_x = (eta[1:] - eta[:-1])/dx

        Un = fs_velocity[:, 0]
        Wn = fs_velocity[:, 1]

        grad_phi = Un + Wn * eta_x

        # Compute eta pseudo-time step
        eta_new = eta[1:] + dt*(-eta_x*grad_phi + Wn*(1+ eta_x**2))

        # Add the last point
        first_eta_val = 1#((self.ylim[1]) * (self.xlim[1]-self.xlim[0]) - (trapezoid(eta_new,x[1:]))) * 2/(x[0]-x[1]) - eta_new[0]
        print("The height at the beginning of the boundary is: ",first_eta_val)
        eta_new = np.append(eta_new[::-1], first_eta_val)[::-1]
        residual = np.linalg.norm(eta-eta_new, np.inf)
        self.etas.append(eta_new)
        return eta_new, residual
    

    def __init_PhiTilde__(self, model) -> None:
        # u = np.array(model.velocity.at(self.fs_points))[:,0]
        # self.PhiTilde = np.zeros_like(self.fs_points[:,0])
        # for i in range(1,len(self.fs_points[:,0])):
        #     self.PhiTilde[i] = trapezoid(u[:i+1],self.fs_points[:i+1,0]) - trapezoid(u[:i],self.fs_points[:i,0])
        self.PhiTilde = np.array(model.init_phi.at(self.fs_points))
        return None
    
    def __compute_phi_tilde(self, model) -> None:
        x = self.fs_points[:,0]
        fs_velocity = np.array(model.velocity.at(self.fs_points))[1:,:2]
        eta = self.fs_points[:, 1]
        dt = self.dt

        # Compute the x-distance between all nodes
        dx = x[1:] - x[:-1] #dx = x_{i+1} - x_{i}

        # First order x-stencil for eta_x
        eta_x = (eta[1:] - eta[:-1])/dx

        Un = fs_velocity[:, 0]
        Wn = fs_velocity[:, 1]

        grad_phi = Un + Wn * eta_x
        

        # Compute eta pseudo-time step
        g = 9.81
        self.PhiTilde[1:] = self.PhiTilde[1:] + dt*(-g*eta[1:] - 0.5*(grad_phi**2 + Wn**2*(1 + eta_x**2)))
        return None
    
    def __check_status__(self, residuals, iter, iter_time, solve_time) -> bool:
        if residuals < self.kwargs.get("rtol", 1e-5):
            print("\n ============================")
            print(" Fs converged")
            print(f" residuals norm {np.linalg.norm(residuals)} after {iter} iterations")
            print(f" Total solve time: {time() - solve_time}")
            print("============================\n")
            return True
        elif residuals > 10000:
            print("\n ============================")
            print(" Fs diverged")
            print(f" residuals norm {np.linalg.norm(residuals)} after {iter} iterations")
            print(f" Total solve time: {time() - solve_time}")
            print("============================\n")
            return True
        else:
            print(f"\t iteration: {iter+1}")
            print(f"\t residual norm {residuals}")
            print(f"\t iteration time: {time() - iter_time}\n")
            return False
        

        
    
    def __update_mesh_data(self, old_eta : np.ndarray, new_eta : np.ndarray) -> None:
        new_mesh = shift_surface(self.mesh, interp1d(self.fs_xs, old_eta), interp1d(self.fs_xs, new_eta))
        self.mesh = new_mesh
        self.fd_mesh.coordinates.dat.data[:] = meshio_to_fd(self.mesh).coordinates.dat.data
        self.V = fd.FunctionSpace(self.fd_mesh, "CG", self.P)
        self.W = fd.VectorFunctionSpace(self.fd_mesh, "CG", self.P)
        fs_indecies = self.V.boundary_nodes(self.kwargs.get("fs", 4))
        self.fs_points = (fd.Function(self.W).interpolate(self.fd_mesh.coordinates).dat.data)[fs_indecies,:]
        self.fs_points = self.fs_points[self.fs_points[:,0].argsort()]
        self.fs_xs = self.fs_points[:,0]
        return None




if __name__ == "__main__":
    kwargs = {"ylim":[-4,1], "V_inf": 10, "g_div": 5, "write":True,
           "n_airfoil": 200,
           "n_fs": 500,
           "n_bed": 20,
           "n_inlet": 15,
           "n_outlet": 15, "a":1, "b":1}
    model = FsSolver("0012" , alpha=10, P=2, kwargs = kwargs)
    model.solve()

if __name__ == "__main__":
    print("yay")