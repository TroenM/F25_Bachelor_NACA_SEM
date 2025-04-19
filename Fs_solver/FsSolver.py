import os
if os.getcwd().split("/")[-1] == "Fs_solver":
    os.chdir("../")
elif os.getcwd() == "/root/MyProjects":
    os.chdir("./F25_Bachelor_NACA_SEM")
 
import sys
sys.path.append(os.getcwd())
currdir = os.getcwd()

import firedrake as fd
from Meshing.mesh_library import *
from Potential_flow_solver.PotentialFlowSolverCLS.PotentialFlowSolver import PotentialFlowSolver
from scipy.interpolate import interp1d
import shutil
from time import time

os.chdir(currdir)

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
        
        self.meshes = [self.mesh]
        self.fs = []
        self.etas = []

        if self.kwargs.get("write", True):
            self.test_output = fd.VTKFile("./Visualisation/FS/test_output.pvd")


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
        solver_time = time()
        print("Solving initial potential flow...")
        PotentialFlow = PotentialFlowSolver(airfoil=self.airfoil, P = self.P, alpha = self.alpha, kwargs = self.kwargs["PFS_kwargs"])
        PotentialFlow.solve()

        if self.kwargs.get("write", True):
            self.test_output.write(PotentialFlow.velocity)
        
        # Updating the free surface and potential at the free surface
        self.old_free_surface_coords = self.__get_free_surface__()
        self.eta = self.old_free_surface_coords[:,1]
        self.__update_free_surface__(model = PotentialFlow)


        # Main Loop
        for iter in range(self.kwargs.get("max_iter", 20)):
            print(f"Iteration {iter+1}/{self.kwargs.get('max_iter', 20)}")
            iter_time = time()

            # Potential flow
            PotentialFlow = PotentialFlowSolver(airfoil=self.airfoil, P = self.P, alpha = self.alpha, kwargs = self.kwargs["PFS_kwargs"])
            PotentialFlow.solve()
            self.__update_free_surface__(model = PotentialFlow)

            if self.kwargs.get("write", True):
                self.test_output.write(PotentialFlow.velocity)

            if self.__check_convergence__(np.max(np.abs(self.eta - self.old_eta)), iter, iter_time, solver_time):
                break
    
    def __update_free_surface__(self, model: PotentialFlowSolver) -> None:
        """
        Updates and sets all variables for the free surface.

        Updates
        ---
        - old_free_surface_coords: coordinates of the current free surface
        - free_surface_coords: coordinates of the free surface after the update
        - free_surface_potential: potential at the free surface
        - mesh: mesh of the entire domain
        - fd_mesh: mesh of the entire domain in firedrake
        """

        fs_coords = self.__get_free_surface__()
        self.old_eta = self.eta.copy()

        self.eta = self.__compute_eta__(model, fs_coords)
        self.phi = self.__compute_phi__(model, fs_coords)

        new_mesh = shift_surface(self.mesh, self.__linear_interpolation__(fs_coords),
                                  self.__linear_interpolation__(np.vstack((fs_coords[:,0], self.eta)).T))
        #self.old_free_surface_coords = fs_coords.copy()
        self.free_surface = fs_coords.copy()

        self.mesh = new_mesh
        self.meshes.append(new_mesh)
        self.fd_mesh.coordinates.dat.data[:] = meshio_to_fd(new_mesh).coordinates.dat.data
        self.kwargs["PFS_kwargs"]["fd_mesh"].coordinates.dat.data[:] = self.fd_mesh.coordinates.dat.data
        self.kwargs["PFS_kwargs"]["mesh"] = new_mesh
        self.kwargs["PFS_kwargs"]["fs_DBC"] = self.phi

    def __get_free_surface__(self) -> np.ndarray:   

        W = fd.VectorFunctionSpace(self.fd_mesh, "CG", self.P)
        fs_nodes = W.boundary_nodes(self.kwargs["PFS_kwargs"]["fs"])
        coords = fd.Function(W).interpolate(self.kwargs["fd_mesh"].coordinates).dat.data_ro[fs_nodes]

        coords = coords[coords[:,0].argsort()] # ensure fs_coords = [[x_0, y_0], [x_1, y_1], ...]
        self.fs.append(coords)

        return coords


    def __compute_eta__(self, PotentialFlow, free_surface_coords) -> np.ndarray:
        """
        solve the free surface equation
            eta_t = -eta_x * u + w*[1 + (eta_x)^2]
        
        using forward Euler method
        """

        free_surface_coords = free_surface_coords.copy()
        free_surface_coords = free_surface_coords[free_surface_coords[:,0].argsort()] # ensure fs_coords = [[x_0, y_0], [x_1, y_1], ...]

        x = free_surface_coords[:,0]
        eta = free_surface_coords[:,1]
        eta[0] = self.kwargs["ylim"][1] # set the first point to the top of the domain

        # Compute eta_x stencil
        dx = x[1:] - x[:-1] # dx = x_{i} - x_{i-1}
        eta_x = (eta[1:] - eta[:-1])/dx # back wind 1-order FDM stencil

        # Get the velocity at the free surface
        velocity = np.array(PotentialFlow.velocity.at(free_surface_coords[1:,:]))
        u = velocity[:,0]
        w = velocity[:,1]

        nablaPhi = u + w*eta_x

        # Compute the new free surface coordinates
        new_eta = eta[1:] + self.dt * (w*(1 + eta_x**2) - eta_x*nablaPhi) # forward Euler method in pseudo time

        # Handle the boundary condition 
        new_eta = np.concatenate(([eta[0]], new_eta)) # Set the first point to the top of the domain
        self.etas.append(new_eta)

        return new_eta
    
    def __linear_interpolation__(self, new_free_surface_coords) -> fd.Function:

        interpolation_func = interp1d(new_free_surface_coords[:,0], new_free_surface_coords[:,1]) #, kind='linear', fill_value="extrapolate")

        return interpolation_func

    def __compute_phi__(self, PotentialFlow, free_surface_coords) -> np.ndarray:
        """
        Solves the free surface potential equation
            phi_t = -g*eta - 1/2 * [u^2 - w^2 * (1 + (eta_x)^2)]
        """

        free_surface_coords = free_surface_coords.copy()
        free_surface_coords = free_surface_coords[free_surface_coords[:,0].argsort()]
        x = free_surface_coords[:,0]
        eta = free_surface_coords[:,1]

        # Compute eta_x stencil
        dx = x[:-1] - x[1:]
        eta_x = (eta[1:] - eta[:-1])/dx

        # Get the velocity at the free surface
        velocity = np.array(PotentialFlow.velocity.at(free_surface_coords[1:,:]))
        u = velocity[:,0]
        w = velocity[:,1]

        nablaPhi = u + w*eta_x

        # Get the potential at the free surface
        g = 9.81
        new_phi = -g*eta[1:] - 0.5* (nablaPhi**2 - w**2 * (1 + eta_x**2)) # forward Euler method in pseudo time


        # Handle the boundary condition 
        new_phi = np.concatenate(([new_phi[0]], new_phi)) # add the last point to the new phi

        return new_phi

    def __check_convergence__(self, free_surface_change, iter, iter_time, solver_time) -> int:

        if free_surface_change < self.kwargs.get("tolerance", 1e-6):
            print(f"Converged after {iter} iterations.")
            print(f"\t Free surface change: {free_surface_change}")
            print(f"\t Solver time: {time() - solver_time:.2f} s")
            return 1

        elif free_surface_change > self.kwargs.get("divergence_tolerance", 1e+3):
            print(f"Did diverged after {iter} iterations.")
            print(f"\t Free surface change: {free_surface_change}")
            print(f"\t Solver time: {time() - solver_time:.2f} s")
            return -1
            
        elif iter == self.kwargs.get("max_iter", 20) - 1:
            print(f"Did not converge after {iter+1} iterations.")
            print(f"\t Free surface change: {free_surface_change}")
            print(f"\t Solver time: {time() - solver_time:.2f} s")
            return -2
        
        else:
            print(f"\t Free surface change: {free_surface_change:.2e}")
            print(f"\t Iteration time: {time() - iter_time:.2f} s")
            return 0

if __name__ == "__main__":
    kwargs = {
        "dt": 0.1,
        "n_fs": 500,
        "max_iter": 3,
        
        "PFS_kwargs": {
            "g_div": 20,
            "a": 1,
            "b":1,
        }
    }

    FsModel = FsSolver(airfoil = "0012", P = 1, alpha = 10, kwargs = kwargs)
    FsModel.solve()