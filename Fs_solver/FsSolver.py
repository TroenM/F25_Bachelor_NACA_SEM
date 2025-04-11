import os
if os.getcwd().split("/")[-1] == "Fs_solver":
    os.chdir("../")
elif os.getcwd() == "/root/MyProjects":
    os.chdir("./F25_Bachelor_NACA_SEM")
 
import sys
sys.path.append(os.getcwd())

import firedrake as fd
from Meshing.mesh_library import *
from Potential_flow_solver.PotentialFlowSolverCLS.PotentialFlowSolver import PotentialFlowSolver
from scipy.interpolate import interp1d
import shutil
from time import time

class FsSolver:
    """
    SEM solver for free surface potential flow around an airfoil.
    """

    def __init__(self, airfoil : str = "0012", P : int = 1, alpha : float = 0, **kwargs):
        """ 
        Initializes the FsSolver
        """
        self.kwargs = kwargs["kwargs"] if kwargs != None else {}

        self.airfoil = airfoil
        self.P = P
        self.alpha = alpha
        self.dt = kwargs.get("dt", 0.01)
        self.mesh = naca_mesh(airfoil, alpha, kwargs = self.kwargs)
        self.fd_mesh = meshio_to_fd(self.mesh)
        self.kwargs["fd_mesh"] = self.fd_mesh

        self.__set_kwargs__()

        return None

    def __set_kwargs__(self):
        """
        Method to set all necessary kwargs for the solver.
        """

        default_kwargs = {
            # Mesh kwargs
            "xlim": [-7, 13],
            "ylim": [-4, 1],
            "n_in": 50,
            "n_out": 50,
            "n_bed": 70,
            "n_fs": 100,
            "n_airfoil": 200,
            "center_of_airfoil": [0, 0.5],


            # Solver kwargs
            "tolerance": 1e-6,
            "divergence_tolerance": 1e+3,
            "max_iter": 20,
            "dt": 0.01,
            }

        for key, value in default_kwargs.items():
            self.kwargs[key] = self.kwargs.get(key, value)
        
        PFS_kwargs = {
            # Mesh
            "mesh": self.mesh,
            "fd_mesh": self.fd_mesh,
            "write": False,

            "xlim": self.kwargs["xlim"],
            "ylim": self.kwargs["ylim"],
            "n_in": self.kwargs["n_in"],
            "n_out": self.kwargs["n_out"],
            "n_bed": self.kwargs["n_bed"],
            "n_fs": self.kwargs["n_fs"],
            "n_airfoil": self.kwargs["n_airfoil"],
            "center_of_airfoil": self.kwargs["center_of_airfoil"],
            "center_of_vortex": self.kwargs["center_of_airfoil"],

            # Boundaries
            "inlet": 1,
            "outlet": 2,
            "bed": 3,
            "fs": 4,
            "naca": 5,

            # Solver
            "solver_params": {"ksp_type": "preonly", "pc_type": "lu"},
            "ksp_rtol": 1e-14,
            "min_tol": 1e-14,
            "dot_tol": 1e-6,
            "gamma_tol": 1e-6,
            "g_div": 7,
            "max_iter": 20,
        }

        self.kwargs["PFS_kwargs"] = self.kwargs["PFS_kwargs"] if "PFS_kwargs" in self.kwargs else {}

        for key, value in PFS_kwargs.items():
            self.kwargs["PFS_kwargs"][key] = self.kwargs["PFS_kwargs"].get(key, value)
    
    def solve(self):
        """
        Solves the potential flow problem.
        """

        # Initial potential flow
        PotentialFlow = PotentialFlowSolver(airfoil=self.airfoil, P = self.P, alpha = self.alpha, kwargs = self.kwargs["PFS_kwargs"])
        PotentialFlow.solve()

        # Updating the free surface and potential at the free surface
        free_surface_coords = self.__get_free_surface__()
        self.new_free_surface_coords = free_surface_coords.copy()
        self.new_free_surface_coords[:,1] = self.__update_free_surface__(PotentialFlow, free_surface_coords)
        self.new_free_surface_func = lambda x: self.__linear_interpolation__(self.new_free_surface_coords)(x)
        

        self.mesh = shift_surface(self.mesh, lambda x: self.kwargs.get("ylim", [-4,1])[1], self.new_free_surface_func)
        self.fd_mesh = meshio_to_fd(self.mesh)
        self.kwargs["PFS_kwargs"]["fd_mesh"] = self.fd_mesh

        new_free_surface_potential = self.__update_free_surface_potential__(PotentialFlow, free_surface_coords)

        # Main Loop
        solver_time = time()
        for iter in range(self.kwargs.get("max_iter", 20)):
            print(f"Iteration {iter+1}/{self.kwargs.get('max_iter', 20)}")
            iter_time = time()

            # Potential flow
            self.kwargs["PFS_kwargs"]["fs_DBC"] = new_free_surface_potential
            PotentialFlow = PotentialFlowSolver(airfoil=self.airfoil, P = self.P, alpha = self.alpha, kwargs = self.kwargs["PFS_kwargs"])
            PotentialFlow.solve()

            # Updating the free surface and potential at the free surface
            old_free_surface_coords = self.new_free_surface_coords

            self.new_free_surface_coords = self.__get_free_surface__()
            self.new_free_surface_coords[:,1] = self.__update_free_surface__(PotentialFlow, old_free_surface_coords)
            free_surface_change = np.linalg.norm(self.new_free_surface_coords - old_free_surface_coords, np.inf)

            if self.__check_convergence__(free_surface_change, iter, solver_time) == True:
                break
            
            self.old_free_surface_func = lambda x: self.new_free_surface_func(x)
            self.new_free_surface_func = lambda x: self.__linear_interpolation__(self.new_free_surface_coords)(x)
            self.mesh = shift_surface(self.mesh, self.old_free_surface_func, self.new_free_surface_func)
            self.fd_mesh = meshio_to_fd(self.mesh)
            self.kwargs["PFS_kwargs"]["fd_mesh"] = self.fd_mesh

            # Updating the free surface potential
            new_free_surface_potential = self.__update_free_surface_potential__(PotentialFlow, old_free_surface_coords)

            print(f"\t Iteration time: {time() - iter_time:.2f} s")
            print(f"\t Free surface change: {free_surface_change:.2e}")
    
    def __get_free_surface__(self) -> np.ndarray:   

        W = fd.VectorFunctionSpace(self.fd_mesh, "CG", self.P)
        fs_nodes = W.boundary_nodes(self.kwargs["PFS_kwargs"]["fs"])
        coords = fd.Function(W).interpolate(self.kwargs["fd_mesh"].coordinates).dat.data_ro[fs_nodes]

        coords = coords[coords[:,0].argsort()] # ensure fs_coords = [[x_0, y_0], [x_1, y_1], ...]

        return coords


    def __update_free_surface__(self, PotentialFlow, free_surface_coords) -> np.ndarray:
        """
        solve the free surface equation
            eta_t = -eta_x * u + w*[1 + (eta_x)^2]
        
        using forward Euler method
        """

        free_surface_coords = free_surface_coords.copy()
        free_surface_coords = free_surface_coords[free_surface_coords[:,0].argsort()] # ensure fs_coords = [[x_0, y_0], [x_1, y_1], ...]

        x = free_surface_coords[:,0]
        eta = free_surface_coords[:,1]

        # Compute eta_x stencil
        dx = x[1:] - x[:-1] # dx = x_{i+1} - x_{i}
        eta_x = eta_x = (eta[1:] - eta[:-1])/dx

        # Get the velocity at the free surface
        velocity = np.array(PotentialFlow.velocity.at(free_surface_coords[:-1,:]))
        u = velocity[:,0]
        w = velocity[:,1]

        # Compute the new free surface coordinates
        new_eta = eta[:-1] + self.dt * (eta_x * u + w + w*eta_x**2) 

        # Handle the boundary condition 
        #########################################################################
        ########### MAYBE SWITCH TO BACKWARD EULER TO FLIP BOUNDARY #############
        #########################################################################
        new_eta = np.concatenate((new_eta, [new_eta[-1]])) # add the last point to the new eta

        return new_eta
    
    def __linear_interpolation__(self, new_free_surface_coords) -> fd.Function:

        interpolation_func = interp1d(new_free_surface_coords[:,0], new_free_surface_coords[:,1], kind='linear', fill_value="extrapolate")

        return interpolation_func

    def __update_free_surface_potential__(self, PotentialFlow, free_surface_coords) -> np.ndarray:
        """
        Solves the free surface potential equation
            phi_t = -g*eta - 1/2 * [u^2 - w^2 * (1 + (eta_x)^2)]
        """

        free_surface_coords = free_surface_coords.copy()
        free_surface_coords = free_surface_coords[free_surface_coords[:,0].argsort()]
        x = free_surface_coords[:,0]
        eta = free_surface_coords[:,1]

        # Compute eta_x stencil
        dx = x[1:] - x[:-1]
        eta_x = (eta[1:] - eta[:-1])/dx

        # Get the velocity at the free surface
        velocity = np.array(PotentialFlow.velocity.at(free_surface_coords[:-1,:]))
        u = velocity[:,0]
        w = velocity[:,1]

        # Get the potential at the free surface
        g = 9.81
        new_phi = -g*eta[:-1] - 1/2 * (u**2 - w**2 * (1 + eta_x**2))


        # Handle the boundary condition 
        #########################################################################
        ########### MAYBE SWITCH TO BACKWARD EULER TO FLIP BOUNDARY #############
        #########################################################################
        new_phi = np.concatenate((new_phi, [new_phi[-1]])) # add the last point to the new phi

        return new_phi

    def __check_convergence__(self, free_surface_change, iter, solver_time):

        if free_surface_change < self.kwargs.get("tolerance", 1e-6):
            print(f"Converged after {iter} iterations.")
            print(f"\t Free surface change: {free_surface_change}")
            print(f"\t Solver time: {time() - solver_time:.2f} s")
            return True

        elif free_surface_change > self.kwargs.get("divergence_tolerance", 1e+3):
            print(f"Did diverged after {iter} iterations.")
            print(f"\t Free surface change: {free_surface_change}")
            return True
            
        elif iter == self.kwargs.get("max_iter", 20) - 1:
            print(f"Did not converge after {iter} iterations.")
            print(f"\t Free surface change: {free_surface_change}")

if __name__ == "__main__":
    kwargs = {
        "PFS_kwargs":{
            "a": 1,
            "b": 1
        }
    }

    FsModel = FsSolver(airfoil = "0012", P = 2, alpha = 10, kwargs = kwargs)
    FsModel.solve()