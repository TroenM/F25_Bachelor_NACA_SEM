import os
if os.getcwd().split("/")[-1] == "Fs_solver":
    os.chdir("../")
elif os.getcwd() == "/root/MyProjects":
    os.chdir("./F25_Bachelor_NACA_SEM")
elif os.getcwd() == "/home/firedrake/Projects":
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
        # Initialize values for solver
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
                              n_in = self.kwargs.get("n_in"),
                              n_out = self.kwargs.get("n_out"))

        self.a = self.kwargs.get("a", 1)
        self.b = self.kwargs.get("b", int(self.airfoil[2:])/100)

        self.dt = self.kwargs.get("dt", 0.001)
        self.fs_rtol = kwargs.get("fs_rtol", 1e-5)

        self.visualisationpath = "../Visualisation/FS/"

        # Create relevant firedrake meshes and functionspaces
        self.fd_mesh = meshio_to_fd(self.mesh)
        self.V = fd.FunctionSpace(self.fd_mesh, "CG", self.P)
        print(f"dof: {self.V.dof_count}")
        self.W = fd.VectorFunctionSpace(self.fd_mesh, "CG", self.P)

        # Find points at free surfac
        fs_indecies = self.V.boundary_nodes(self.kwargs.get("fs", 4))
        self.fs_points = (fd.Function(self.W).interpolate(self.fd_mesh.coordinates).dat.data)[fs_indecies,:]
        self.fs_sorted = self.fs_points[np.argsort(self.fs_points[:,0])]
        self.fs_xs = self.fs_sorted[:,0]

        self.etas = np.zeros((self.kwargs.get("max_iter_fs", 10), len(self.fs_xs)))
        self.phis = np.zeros((self.kwargs.get("max_iter_fs", 10), len(self.fs_xs)))
        self.fs_xs_array = np.zeros((self.kwargs.get("max_iter_fs", 10), len(self.fs_xs)))
        self.residual_array = np.zeros(self.kwargs.get("max_iter_fs", 10))
        
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
        old_eta = self.fs_sorted[:,1]
        # new_eta = self.__init_mesh_guess__()
        # self.__update_mesh_data__(old_eta, new_eta)
        model = PotentialFlowSolver(self.airfoil , self.P, self.alpha, kwargs=kwargs_for_Kutta_kondition)
        model.solve()
        self.model = model
        self.velocity = model.velocity

        if self.write:
            self.velocity_output.write(self.velocity)

        # Initialize phi tilde
        self.__init_PhiTilde__(model)
        

        # Preparing for loop
        
        print("initialization done")
        # Start loop for iterating free surface
        for i in range(self.kwargs.get("max_iter_fs", 10)):
            # Start iteration time
            iter_time = time()
            
            # Update eta and dirichlet boundary condition
            new_eta, self.PhiTilde, residuals = self.__compute_fs_equations_weak1d__()
            kwargs_for_Kutta_kondition["fs_DBC"] = self.PhiTilde
            kwargs_for_Kutta_kondition["fs_xs"] = self.fs_xs

            # Update mesh with new eta
            self.__update_mesh_data__(old_eta, new_eta)
            kwargs_for_Kutta_kondition["mesh"] = self.mesh
            kwargs_for_Kutta_kondition["fd_mesh"] = self.fd_mesh

            self.etas[i, :] = new_eta.copy()
            self.phis[i, :] = self.PhiTilde.copy()
            self.fs_xs_array[i, :] = self.fs_xs.copy()
            self.residual_array[i] = residuals.copy()

            # Solve model and kutta kondition again with new condition at the free surface
            model = PotentialFlowSolver(self.airfoil , self.P, self.alpha, kwargs=kwargs_for_Kutta_kondition)
            model.solve()

            
            # Itterate solver by deeming eta^n+1 to eta^n instead
            old_eta = new_eta.copy()

            # Save velocity within the solver and write it
            self.velocity = model.velocity
            if self.write:
                self.velocity_output.write(self.velocity)
            
            # If solver is converged or diverged to a certain extend, stop solver
            if self.__check_status__(residuals,i,iter_time,solve_time):
                break
        
        return None

    def __init_PhiTilde__(self, model) -> None:
        # Initialize phi tilde
        self.PhiTilde = np.array(model.init_phi.at(self.fs_sorted))
        return None
    
    def __init_mesh_guess__(self) -> np.ndarray:
        """
        Initialize the mesh guess for the free surface
        """
        x = self.fs_sorted[:,0]
        eta = self.fs_sorted[:,1]
        # omega = 2*np.pi/(2)
        # mask = (x >= 0)*(x <= 2)
        # eta[mask] = np.sin(omega*(x[mask]))/100 + self.ylim[1]

        return eta


    def __compute_fs_equations_weak1d__(self) -> None:
        """
        Updates the free surface by solving the free surface equations using firedrake
        """
        # Mesh and function spaces
        number_of_points = self.fs_sorted.shape[0]
        fs_mesh = fd.IntervalMesh(number_of_points-1, self.xlim[0], self.xlim[1])
        fs_mesh.coordinates.dat.data[:] = self.fs_sorted[:,0] # Setting coordinats to match actual points

        V_eta = fd.FunctionSpace(fs_mesh, "CG", 1)
        V_phi = fd.FunctionSpace(fs_mesh, "CG", 1)

        # Defining unknown functions
        W = V_eta * V_phi
        fs_vars = fd.Function(W)
        eta_n1, phi_n1 = fd.split(fs_vars) #eta^{n+1}, phi^{n+1}
        v_1, v_2 = fd.TestFunctions(W) 

        # Defining known functions
        eta_n = fd.Function(V_eta) # eta^{n}
        phi_n = fd.Function(V_phi) # phi^{n}
        u_n = fd.Function(V_phi) # u^{n}
        w_n = fd.Function(V_phi) # w^{n}
        velocity = np.array(self.velocity.at(self.fs_sorted)) # velocity at free surface points

        phi_n.dat.data[:] = self.PhiTilde
        eta_n.dat.data[:] = self.fs_sorted[:, 1] - self.ylim[1] 
        u_n.dat.data[:] = velocity[:, 0]
        w_n.dat.data[:] = velocity[:, 1]

        g = fd.Constant(9.81)
        dt = fd.Constant(self.dt)

        # Constants relevant for dampening
        xd_in = fd.Constant(self.kwargs.get("xd_in",-4.0))
        xd_out = fd.Constant(self.kwargs.get("xd_out", 10))
        x = fd.SpatialCoordinate(fs_mesh)[0]
        A = fd.Constant(self.kwargs.get("damp", 100))

        # Dampen eta towards the "normal" height of the domain at the edges
        eta_damp_in = A*fd.conditional(x < xd_in, ((x - xd_in) / (self.xlim[0]  - xd_in))**2, 0)*eta_n1
        eta_damp_out = A*fd.conditional(x > xd_out, ((x - xd_out) / (self.xlim[1] - xd_out))**2, 0)*eta_n1

        bcs_eta = fd.DirichletBC(W.sub(0), 0, "on_boundary") # Dirichlet BC for eta
        #bcs_phi1 = fd.DirichletBC(W.sub(1), self.PhiTilde[0], 1) # Dirichlet BC for phi
        bcs_phi2 = fd.DirichletBC(W.sub(1), self.PhiTilde[-1], 2) # Dirichlet BC for phi

        bcs = [bcs_eta]

        a1 = fd.inner(eta_n1 - eta_n, v_1)*fd.dx + fd.inner(eta_damp_in, v_1)*fd.dx + fd.inner(eta_damp_out, v_1)*fd.dx
        #L1 = dt*(fd.inner(w_n, v_1)*fd.dx - fd.inner(u_n*eta_n1.dx(0), v_1)*fd.dx)
        L1 = dt * (-fd.inner(fd.dot(eta_n1.dx(0),phi_n1.dx(0)),v_1)*fd.dx + 
                   fd.inner(w_n * (fd.Constant(1) + fd.dot(eta_n1.dx(0), eta_n1.dx(0))), v_1)*fd.dx)
        F1 = a1 - L1

        a2 = fd.inner(phi_n1 - phi_n, v_2)*fd.dx
        L2 = dt*(-fd.inner(g*eta_n1, v_2)*fd.dx -
            fd.Constant(0.5)*fd.inner(phi_n1.dx(0)**2, v_2)*fd.dx + 
            fd.Constant(0.5)*fd.inner(w_n**2 * (fd.Constant(1) + eta_n1.dx(0)**2), v_2)*fd.dx)
        F2 = a2 - L2

        F = F1 + F2

        # solver_params={'ksp_type': 'gmres',
                    #   'pc_type': 'hypre',
                    #    'ksp_max_it': 100}

        solver_params = {
             "newton_solver": {
                 "relative_tolerance": self.fs_rtol,
                 "absolute_tolerance": 1e-8,  # Add tighter absolute tolerance if needed
                 "maximum_iterations": 500,   # Increase maximum iterations
                 "relaxation_parameter": 1.0  # Adjust relaxation if needed
             }
        }
        fd.solve(F == 0, fs_vars, bcs = bcs, solver_parameters=solver_params)

        eta_new = fs_vars.sub(0) 
        phi_new = fs_vars.sub(1)
        eta_new = np.array(eta_new.at(self.fs_xs)) + self.ylim[1]
        phi_new = np.array(phi_new.at(self.fs_xs))

        old_eta = self.fs_sorted[:, 1]
        
        sort_mask = np.argsort(self.fs_xs)
        eta_new = eta_new[sort_mask]
        phi_new = phi_new[sort_mask]
        old_eta = old_eta[sort_mask]
        self.fs_xs.sort()

        residuals = np.linalg.norm(eta_new - old_eta, np.inf)

        return eta_new, phi_new, residuals

    def __check_status__(self, residuals, iter, iter_time, solve_time) -> bool:
        # If convergence kriteria is met print relevant information
        if residuals < self.kwargs.get("fs_rtol", 1e-5):
            print("\n ============================")
            print(" Fs converged")
            print(f" residuals norm {np.linalg.norm(residuals)} after {iter} iterations")
            print(f" Total solve time: {time() - solve_time}")
            print("============================\n")
            return True
        # If divergence kriteria is met print relevant information
        elif residuals > 10000:
            print("\n ============================")
            print(" Fs diverged")
            print(f" residuals norm {np.linalg.norm(residuals)} after {iter} iterations")
            print(f" Total solve time: {time() - solve_time}")
            print("============================\n")
            return True
        # If the maximum amout of iterations is done print relevant information
        elif iter >= self.kwargs.get("max_iter_fs", 10) - 1:
            print("\n ============================")
            print(" Fs did not converge")
            print(f" residuals norm {np.linalg.norm(residuals)} after {iter} iterations")
            print(f" Total solve time: {time() - solve_time}")
            print("============================\n")
            return True
        # If none of the above, print relevant information about solver status
        else:
            print(f"\t iteration: {iter+1}")
            print(f"\t residual norm {residuals}")
            print(f"\t iteration time: {time() - iter_time}\n")
            return False          
    
    def __update_mesh_data__(self, old_eta : np.ndarray, new_eta : np.ndarray) -> None:
        # Shift surface of the mesh and set this as new mesh

        old_eta_sorted = old_eta
        new_eta_sorted = new_eta

        func_before = interp1d(self.fs_xs, old_eta_sorted)
        func_after = interp1d(self.fs_xs, new_eta_sorted)

        self.fd_mesh.coordinates.dat.data[:] = shift_surface(self.fd_mesh, func_before, func_after).coordinates.dat.data

        # Change the firedrake function spaces to match the new mesh
        self.V = fd.FunctionSpace(self.fd_mesh, "CG", self.P)
        self.W = fd.VectorFunctionSpace(self.fd_mesh, "CG", self.P)

        # Find points at free surface
        fs_indecies = self.V.boundary_nodes(self.kwargs.get("fs", 4))
        self.fs_points = (fd.Function(self.W).interpolate(self.fd_mesh.coordinates).dat.data)[fs_indecies,:]
        self.fs_sorted = self.fs_points[np.argsort(self.fs_points[:,0])]
        self.fs_xs = self.fs_sorted[:,0]

        return None




if __name__ == "__main__":
    kwargs = {"ylim":[-2,1], "xlim":[-6,12], 
            "xd_in": -3, "xd_out": 10,

            "V_inf": 50, 
            "g_div": 7, 
            "write":True,
            "n_airfoil": 550,
            "n_fs": 350,
            "n_bed": 70,
            "n_in": 30,
            "n_out": 30,
            "rtol": 1e-8,
            "a":1, "b":1,
            "max_iter": 50,
            "dot_tol": 1e-4,

            "fs_rtol": 1e-6,
            "max_iter_fs":3,
            
            "dt": 5e-3,
            "damp":200}
    
    FS = FsSolver("0012", alpha = 5, P=1, kwargs = kwargs)
    FS.solve()
    etas = np.array(FS.etas)
    phis = np.array(FS.phis)
    fs_xs = np.array(FS.fs_xs_array)
    residuals = np.array(FS.residual_array)

    print(os.getcwd(), "\n")
    np.savetxt("./results/eta.txt", etas)
    np.savetxt("./results/phiTilde.txt", phis)
    np.savetxt("./results/fs_xs.txt", fs_xs)
    np.savetxt("./results/residuals.txt", residuals)