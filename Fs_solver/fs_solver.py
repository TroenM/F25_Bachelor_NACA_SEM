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
        # self.fs_points = self.fs_points[self.fs_points[:,0].argsort()]
        self.fs_xs = self.fs_points[:,0]

        self.etas = []

        self.a = self.kwargs.get("a", 1)
        self.b = self.kwargs.get("b", int(self.airfoil[2:])/100)

        self.dt = self.kwargs.get("dt", 0.001)
        self.fs_rtol = kwargs.get("fs_rtol", 1e-5)

        self.visualisationpath = "../Visualisation/FS/"
        self.damping = self.kwargs.get("damping", 1)
        
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
        old_eta = self.fs_points[:,1]
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
        self.init_phi = self.PhiTilde.copy() # Remove after debug
        

        # Preparing for loop
        
        print("initialization done")
        for i in range(self.kwargs.get("max_iter_fs", 10)):
            iter_time = time()
            
            # Update free surface
            #new_eta, residuals = self.__compute_eta__(model)

            # Update dirichlet boundary condition
            #self.__compute_phi_tilde__(model, new_eta)

            new_eta, self.PhiTilde, residuals = self.__compute_fs_equations_weak1d__()
            kwargs_for_Kutta_kondition["fs_DBC"] = self.PhiTilde

            # Update mesh
            self.__update_mesh_data__(old_eta, new_eta)
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

            

    def __compute_eta__(self, model : PotentialFlowSolver) -> np.ndarray:
        x = self.fs_points[:,0]
        fs_velocity = np.array(model.velocity.at(self.fs_points))[1:]
        eta = self.fs_points[:, 1]
        dt = self.dt

        # Compute the x-distance between all nodes
        dx = x[1:] - x[:-1] #dx = x_{i} - x_{i-1}

        # First order x-stencil for eta_x
        eta_x = (eta[1:] - eta[:-1])/dx

        Wn = fs_velocity[:, 1]

        grad_phi_tilde = (self.PhiTilde[1:] - self.PhiTilde[:-1])/dx

        # Compute eta pseudo-time step
        eta_new = eta[1:] + dt*(-eta_x*grad_phi_tilde + Wn*(1 + eta_x**2)*self.damping)
        
        # Add the last point
        first_eta_val = self.ylim[1]#((self.ylim[1]) * (self.xlim[1]-self.xlim[0]) - (trapezoid(eta_new,x[1:]))) * 2/(x[0]-x[1]) - eta_new[0]
        print("The height at the beginning of the boundary is: ",first_eta_val)
        eta_new = np.hstack((first_eta_val, eta_new))
        residual = np.linalg.norm(eta-eta_new, np.inf)
        self.etas.append(eta_new)
        return eta_new, residual

    def __init_PhiTilde__(self, model) -> None:
        self.PhiTilde = np.array(model.init_phi.at(self.fs_points))
        return None
    
    def __init_mesh_guess__(self) -> np.ndarray:
        """
        Initialize the mesh guess for the free surface
        """
        x = self.fs_points[:,0]
        eta = self.fs_points[:,1]
        omega = 2*np.pi/(2)
        mask = (x >= 0)*(x <= 2)
        eta[mask] = np.sin(omega*(x[mask]))/100 + self.ylim[1]

        return eta

    def __compute_phi_tilde__(self, model, eta) -> None:
        x = self.fs_points[:,0]
        fs_velocity = np.array(model.velocity.at(self.fs_points))[1:,:2]
        dt = self.dt

        # Compute the x-distance between all nodes
        dx = x[1:] - x[:-1] #dx = x_{i+1} - x_{i}

        # First order x-stencil for eta_x
        eta_x = (eta[1:] - eta[:-1])/dx

        Wn = fs_velocity[:, 1]

        grad_phi_tilde = (self.PhiTilde[1:] - self.PhiTilde[:-1])/dx
        
        

        # Compute eta pseudo-time step
        g = 9.81
        self.PhiTilde[1:] = self.PhiTilde[1:] + dt*(-g*eta[1:] - 0.5*(grad_phi_tilde**2*self.damping - Wn**2*(1 + eta_x**2)*self.damping))
        return None


#    def __compute_fs_equations_weak1d__(self) -> None:
        """
        Updates the free surface by solving the free surface equations using firedrake
        """
        print("\t Computing free surface equations using IE")
        # Mesh and function spaces
        number_of_points = self.fs_points.shape[0]
        fs_mesh = fd.IntervalMesh(number_of_points-1, self.xlim[0], self.xlim[1])
        fs_mesh.coordinates.dat.data[:] = self.fs_points[:,0] # Setting coordinats to match actual points

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
        velocity = np.array(self.velocity.at(self.fs_points)) # velocity at free surface points

        phi_n.dat.data[:] = self.PhiTilde
        eta_n.dat.data[:] = self.fs_points[:, 1] - self.ylim[1] 
        u_n.dat.data[:] = velocity[:, 0]
        w_n.dat.data[:] = velocity[:, 1]

        g = fd.Constant(9.81)
        dt = fd.Constant(self.dt)

        bcs_eta = fd.DirichletBC(W.sub(0), 0, "on_boundary") # Dirichlet BC for eta
        bcs_phi1 = fd.DirichletBC(W.sub(1), self.PhiTilde[0], 1) # Dirichlet BC for phi
        bcs_phi2 = fd.DirichletBC(W.sub(1), self.PhiTilde[-1], 2) # Dirichlet BC for phi

        bcs = [bcs_eta, bcs_phi1]

        F1 = fd.inner(eta_n1 - eta_n, v_1)*fd.dx + dt*(fd.inner(eta_n1.dx(0)*phi_n1.dx(0), v_1)*fd.dx - 
                                                       fd.inner(w_n, v_1)*fd.dx
                                                       )

        F2 = fd.inner(phi_n1 - phi_n, v_2)*fd.dx + dt*(
            fd.inner(g*eta_n1, v_2)*fd.dx +
            fd.Constant(0.5)*fd.inner(phi_n1.dx(0)**2 + w_n**2, v_2)*fd.dx
        )

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

        eta_new = fs_vars.sub(0).dat.data[:] + self.ylim[1]
        phi_new = fs_vars.sub(1).dat.data[:]
        old_eta = self.fs_points[:, 1]
        residuals = np.linalg.norm(eta_new - old_eta, np.inf)

        print(f"\t free surface equations done")

        return eta_new, phi_new, residuals

    def __compute_fs_equations_weak1d__(self) -> tuple:
        """
        Updates the free surface and corresponding dirichlet boundary condition
        """
        #print("\t Computing free surface equations using EE")
        # Initialize mesh
        number_of_points = self.fs_points.shape[0]
        fs_mesh = fd.IntervalMesh(number_of_points-1, self.xlim[0], self.xlim[1])

        # Ensure the mesh coordinates are spaced correctly
        fs_mesh.coordinates.dat.data[:] = self.fs_points[:,0]
        V_eta = fd.FunctionSpace(fs_mesh, "CG", 1)
        V_phi = fd.FunctionSpace(fs_mesh, "CG", 1)

        # Defining unknown functions
        eta_n1 = fd.TrialFunction(V_eta)
        phi_n1 = fd.TrialFunction(V_phi)

        v_eta = fd.TestFunction(V_eta)
        v_phi = fd.TestFunction(V_phi)

        # Defining known functions
        eta_n = fd.Function(V_eta)
        eta_n.dat.data[:] = self.fs_points[:, 1] - self.ylim[1]
        phi_n = fd.Function(V_phi)
        phi_n.dat.data[:] = self.PhiTilde
        u_n = fd.Function(V_phi)
        u_n.dat.data[:] = np.array(self.velocity.at(self.fs_points))[:, 0]
        w_n = fd.Function(V_phi)
        w_n.dat.data[:] = np.array(self.velocity.at(self.fs_points))[:, 1]
        g = fd.Constant(9.82)
        dt = fd.Constant(self.dt)

        V_inf = fd.Constant(self.V_inf)
        xd_in = fd.Constant(-6.0)
        xd_out = fd.Constant(10.5)
        x = fd.SpatialCoordinate(fs_mesh)[0]
        A = fd.Constant(100)
        eta_damp_in = A*fd.conditional(x < xd_in, ((x - xd_in) / (self.xlim[0]  - xd_in))**2, 0)*eta_n1
        eta_damp_out = A*fd.conditional(x > xd_out, ((x - xd_out) / (self.xlim[1] - xd_out))**2, 0)*eta_n1
        #phi_damp_in = A*fd.conditional(x < xd_in, ((x - xd_in) / (self.xlim[0]  - xd_in))**2, 0)*phi_n1 \
                     #-A*fd.conditional(x < xd_in, ((x - xd_in) / (self.xlim[0]  - xd_in))**2 * V_inf, 0)
        #phi_damp_out = A*fd.conditional(x > xd_out, ((x - xd_out) / (self.xlim[1] - xd_out))**2, 0)*phi_n1 \
                      #-A*fd.conditional(x > xd_out, ((x - xd_out) / (self.xlim[1] - xd_out))**2 * V_inf, 0)

        # Weak form EE schemes of the updating eta
        F_eta_lhs = fd.inner(eta_n1, v_eta)*fd.dx + fd.inner(eta_damp_in, v_eta)*fd.dx + fd.inner(eta_damp_out, v_eta)*fd.dx
        F_eta_rhs = fd.inner(eta_n, v_eta)*fd.dx + dt * (fd.inner(w_n, v_eta)*fd.dx - fd.inner(u_n*eta_n.dx(0), v_eta)*fd.dx)
        
        dbc_eta = fd.DirichletBC(V_eta, 0, "on_boundary")
        eta_new = fd.Function(V_eta)
        fd.solve(F_eta_lhs == F_eta_rhs, eta_new, bcs = dbc_eta, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

        # Weak form EE schemes of the updating phi
        F_phi_lhs = fd.inner(phi_n1, v_phi)*fd.dx #+ fd.inner(phi_damp_in, v_phi)*fd.dx + fd.inner(phi_damp_out, v_phi)*fd.dx
        F_phi_rhs = fd.inner(phi_n, v_phi)*fd.dx + dt * (-fd.Constant(0.5)*fd.inner(u_n**2 + w_n**2, v_phi)*fd.dx-g*fd.inner(eta_n, v_phi)*fd.dx) + fd.inner((eta_new - eta_n)*w_n, v_phi)* fd.dx
        
        dbc_phi = []# [fd.DirichletBC(V_phi, self.PhiTilde[0], 1)]#, fd.DirichletBC(V_phi, self.PhiTilde[-1], 2)]
        phi_new = fd.Function(V_phi)
        fd.solve(F_phi_lhs == F_phi_rhs, phi_new, bcs = dbc_phi, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

        old_eta = self.fs_points[:, 1]
        eta_new.dat.data[:] += self.ylim[1]
        residuals = np.linalg.norm(eta_new.dat.data[:] - old_eta, np.inf)
        #print(f"\t free surface equations done")

        return eta_new.dat.data[:], phi_new.dat.data[:], residuals

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
        elif iter >= self.kwargs.get("max_iter_fs", 10):
            print("\n ============================")
            print(" Fs did not converge")
            print(f" residuals norm {np.linalg.norm(residuals)} after {iter} iterations")
            print(f" Total solve time: {time() - solve_time}")
            print("============================\n")
            return True
        else:
            print(f"\t iteration: {iter+1}")
            print(f"\t residual norm {residuals}")
            print(f"\t iteration time: {time() - iter_time}\n")
            return False
        

        
    
    def __update_mesh_data__(self, old_eta : np.ndarray, new_eta : np.ndarray) -> None:
        new_mesh = shift_surface(self.mesh, interp1d(self.fs_xs, old_eta), interp1d(self.fs_xs, new_eta))
        self.mesh = new_mesh
        self.fd_mesh.coordinates.dat.data[:] = meshio_to_fd(self.mesh).coordinates.dat.data
        self.V = fd.FunctionSpace(self.fd_mesh, "CG", self.P)
        self.W = fd.VectorFunctionSpace(self.fd_mesh, "CG", self.P)
        fs_indecies = self.V.boundary_nodes(self.kwargs.get("fs", 4))
        self.fs_points = (fd.Function(self.W).interpolate(self.fd_mesh.coordinates).dat.data)[fs_indecies,:]
        # self.fs_points = self.fs_points[self.fs_points[:,0].argsort()]
        self.fs_xs = self.fs_points[:,0]
        return None




if __name__ == "__main__":
    kwargs = {"ylim":[-4,1], "xlim":[-10,13], "V_inf": 10, "g_div": 70, "write":True,
           "n_airfoil": 50,
           "n_fs": 230,
           "n_bed": 20,
           "n_inlet": 10,
           "n_outlet": 10,
           "rtol": 1e-8,
           "fs_rtol": 1e-3,
           "max_iter_fs": 50,
           "max_iter": 50,
           "dt": 1e-2,
           "a":1, "b":1,
           "dot_tol": 1e-4}
    
    FS = FsSolver("0012", alpha = 10, P=3, kwargs = kwargs)
    FS.solve()
