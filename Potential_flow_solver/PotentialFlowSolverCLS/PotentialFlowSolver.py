import numpy as np
import firedrake as fd
import shutil
import os
import sys
from time import time

#### Running from F25_Bachelor_NACA_SEM ####
try:
    os.chdir("./F25_Bachelor_NACA_SEM/")
except:
    pass

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
    bed : int, Default 3
        - Index for the seabed
    fs : int, Default: 4
        - Index for free surface boundary
    naca : int, Default: 5
        - Index for NACA airfoil boundary   

 
    mesh : meshio.Mesh, Default: naca_mesh filled with kwargs
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
        self.alpha = np.deg2rad(alpha)
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

        self.mesh = self.kwargs.get("mesh", naca_mesh(self.airfoil, np.rad2deg(self.alpha), self.xlim, self.ylim, 
                              center_of_airfoil=self.center_of_airfoil,
                              n_airfoil = self.kwargs.get("n_airfoil"),
                              n_fs = self.kwargs.get("n_fs"),
                              n_bed = self.kwargs.get("n_bed"),
                              n_inlet = self.kwargs.get("n_inlet"),
                              n_outlet = self.kwargs.get("n_outlet")))
        self.fd_mesh = self.kwargs.get("fd_mesh", meshio_to_fd(self.mesh))

        self.a = self.kwargs.get("a", 1)
        self.b = self.kwargs.get("b", int(self.airfoil[2:])/100)

        self.solver_params = self.kwargs.get("solver_params", {"ksp_type": "preonly", "pc_type": "lu"})

        # Handeling output files
        if self.write:
            if os.path.exists("./velocity_output"):
                shutil.rmtree("./velocity_output")
            if os.path.exists("./vortex_output"):
                shutil.rmtree("./vortex_output")
            if os.path.exists("./pressure_output"):
                shutil.rmtree("./pressure_output")

            try:
                os.remove("./velocity_output.pvd")
            except:
                pass
            try:
                os.remove("./vortex_output.pvd")
            except:
                pass
            try:
                os.remove("./pressure_output.pvd")
            except:
                pass
            
            self.velocity_output = fd.VTKFile("velocity_output.pvd")
            self.vortex_output = fd.VTKFile("vortex_output.pvd")
            self.BC_output = fd.VTKFile("velocityBC.pvd")


    def solve(self):
        center_of_vortex = self.kwargs.get("center_of_vortex", self.center_of_airfoil)
        # Identify trailing edge and leading edge
        p1, p_te, p_leading_edge, pn= self.get_edge_info()
        v12 = (pn - p1)
        p_te_new = (p_te-center_of_vortex)*(1 + 1e-2) + center_of_vortex
        p1new = p1 - v12 * 0.4 
        pnnew = pn + v12 * 0.4
        # print(p1new)
        # print(pnnew)
        # print(p_te_new)


        # Initializing Laplaze solver
        model = PoissonSolver(self.fd_mesh, P=self.P)
        model.impose_NBC(fd.Constant(-self.V_inf), self.kwargs.get("inlet", 1))
        model.impose_NBC(fd.Constant(self.V_inf), self.kwargs.get("outlet", 2))

        # Free surface boundary condition
        if self.kwargs.get("fs_DBC", np.array([0])).any():
            boundary_indecies = model.V.boundary_nodes(self.kwargs.get("fs", 4))
            print(len((fd.Function(model.W).interpolate(model.mesh.coordinates).dat.data)[:,0]))
            xs = (fd.Function(model.W).interpolate(model.mesh.coordinates).dat.data)[boundary_indecies,0]
            ys = self.kwargs.get("fs_DBC")
            print(len(xs))
            print(len(ys))
            model.impose_DBC(interp1d(xs,ys), self.kwargs.get("fs", 4), "only_x")
        
        converged = False
        while not converged:
            try:
                model.solve(self.solver_params)
                self.solver_params["ksp_rtol"] = self.kwargs.get("min_rtol", 1e-14)
                converged = True
            except:
                self.solver_params["ksp_rtol"] *= 10
                print(f"Solver did not converge, increasing ksp_rtol to {self.solver_params['ksp_rtol']}")


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
        self.Gamma = 0
        Gammas = [0]
        
        # Main loop
        for it, _ in enumerate(range(self.kwargs.get("max_iter", 20) + 1)):
            time_it = time()
            print(f"Starting iteration {it}")

            # Computing the vortex strength
            vte = velocity.at(p_te_new)
            #vte = velocity.at(p1new) + velocity.at(pnnew)
            #Gamma = self.compute_circular_vortex_strength(v12, vte, p_te_new, center_of_vortex) # TO BE IMPLEMENTED
            Gamma = self.compute_vortex_strength(v12, vte, p_te_new)
            Gammas.append(Gamma)
            Gamma /= self.__compute_Gamma_div(Gammas)
            
            Gammas[-1] = Gamma
            self.Gamma += Gamma

            # Compute the vortex
            #vortex = self.compute_circular_vortex(Gamma/20, model, center_of_vortex, vortex)
            vortex = self.compute_vortex(Gamma, model, center_of_vortex, vortex)

            velocity += vortex
            vortex_sum += vortex

            # Computing the boundary correction
            velocityBC = self.compute_boundary_correction(vortex)
            velocity += velocityBC
            velocityBC_sum += velocityBC
            
            # Check for convergence
            if np.abs(np.dot(v12, vte)) < self.kwargs.get("dot_tol", 1e-6):
                print(f"Solver converged in {it} iterations")
                print(f"\t Total time: {time() - time_total}")
                print(f"\t dGamma: {np.abs(Gamma - old_Gamma)}")
                print(f"\t dot product: {np.dot(v12, vte)}")
                print(f"\n")
                break

            # Checking for Stagnation
            if np.abs(Gamma - old_Gamma) < self.kwargs.get("gamma_tol", 1e-6):

                print(f"Solver stagnated in {it-1} iterations")
                print(f"\t Total time: {time() - time_total}")
                print(f"\t dGamma: {np.abs(Gamma - old_Gamma)}")
                print(f"\t dot product: {np.dot(v12, vte)}")
                print(f"\n")
                break

            # Checking for divergence
            if np.abs(Gamma - old_Gamma) > 1e4:
                print(f"Solver diverged in {it} iterations")
                print(f"\t Total time: {time() - time_total}")
                print(f"\t dGamma: {np.abs(Gamma - old_Gamma)}")
                print(f"\t dot product: {np.dot(v12, vte)}")
                print(f"\n")
                break

            # Updating the vortex strength
            print(f"\t dGamma: {Gamma - old_Gamma}")
            print(f"\t dot product: {np.dot(velocity.at(p_te_new), v12)}")
            old_Gamma = Gamma

            # Write to file
            if self.write:
                self.velocity_output.write(velocity)
                self.vortex_output.write(vortex)
                vortex += velocityBC
                self.BC_output.write(vortex)

            print(f"\t Iteration time: {time() - time_it} seconds\n")

            # END OF MAIN LOOP

        self.lift = -self.Gamma * self.V_inf * self.kwargs.get("rho", 1.225)
        self.lift_coeff = self.lift / (1/2 * self.kwargs.get("rho", 1.225) * self.V_inf**2)

        self.__compute_pressure_coefficients(velocity, model)

        self.velocity = velocity

        if it == self.kwargs.get("max_iter", 20):
            print(f"Solver did not converge in {it} iterations")
            print(f"\t Total time: {time() - time_total}")
            print(f"\t dGamma: {np.abs(Gamma - old_Gamma)}")
            print(f"\n")

    def __compute_Gamma_div(self, Gammas : list) -> float:
        if len(Gammas) <4:
            self.g_div = self.kwargs.get("g_div", 7)
            return self.g_div
        else:
            self.g_div = self.g_div*(Gammas[-3] - Gammas[-2] - Gammas[-2] + Gammas[-1]/self.g_div)/(Gammas[-3] - Gammas[-2])
            return self.g_div
        return self.kwargs.get("g_div", 7)

    def get_edge_info(self):
        """
        Returns the coordinates of the leading edge, trailing edge and the point at the trailing edge
        """
        # fetching points on the NACA airfoil
        naca_lines = self.mesh.cells_dict["line"][np.where(
            np.concatenate(self.mesh.cell_data["gmsh:physical"]) == self.kwargs.get("naca", 5))[0]]
        naca_points = np.unique(naca_lines)

        p_te = self.mesh.points[np.min(naca_points)][:2] # Lower point at trailing edge
        p1 = self.mesh.points[np.min(naca_points)+1][:2] # Second lower point at trailing edge

        pn = self.mesh.points[np.max(naca_points)][:2] # Upper point at trailing edge

        # Assuming the leading edge is at (0,0) before rotation
        alpha = self.alpha
        p_leading_edge = np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]]) @ np.array([-1, 0]) + np.array([1, 0])
        return p1, p_te, p_leading_edge, pn

    def compute_vortex_strength(self, v12, vte, p_te_new) -> float:
        """
        Computes the vortex strength for the given iteration
        """
        a = self.a
        b = self.b

        ##################################################
        ###### ALPHA IS SET TO ZERO FOR NOW ##############
        ##################################################
        alpha = self.alpha

        # Get the coordinates of the trailing edge
        p_x = p_te_new[0]
        p_y = p_te_new[1]

        # Translating to center of airfoil
        x_t = p_x - self.center_of_airfoil[0]
        y_t = p_y - self.center_of_airfoil[1]

        # Rotating to align with airfoil
        x_rot = x_t * np.cos(alpha) - y_t * np.sin(alpha)
        y_rot = x_t * np.sin(alpha) + y_t * np.cos(alpha)

        # Scaling to unit circle
        x_s = x_rot / a
        y_s = y_rot / b

        # Computing vortex at trailing edge without Gamma/2pi
        Wx = -y_s / (x_s**2 + y_s**2)
        Wy = x_s / (x_s**2 + y_s**2)

        
        # Rotating back to global coordinates
        Wx_rot = Wx * np.cos(-alpha) - Wy * np.sin(-alpha)
        Wy_rot = Wx * np.sin(-alpha) + Wy * np.cos(-alpha)

        elipse_circumference = np.pi*(3*(a+b)-np.sqrt((3*a+b)*(a+3*b)))

        # Computing the vortex strength
        Gamma = -elipse_circumference*(v12[0]*vte[0] + v12[1]*vte[1])/(Wx_rot*v12[0] + Wy_rot*v12[1])

        return Gamma

    def compute_vortex(self, Gamma, model, center_of_vortex, vortex) -> fd.Function:
        """
        Computes the vortex field for the given vortex strength
        """

        ##################################################
        ###### ALPHA IS SET TO ZERO FOR NOW ##############
        ##################################################

        alpha = self.alpha  # Convert angle of attack to radians
        alpha = fd.Constant(alpha)

        # Extract airfoil scaling parameters
        a = self.a
        b = self.b

        # Translate coordinates to vortex-centered frame
        x_shifted = model.x - center_of_vortex[0]
        y_shifted = model.y - center_of_vortex[1]

        x_rot = x_shifted * fd.cos(alpha) - y_shifted * fd.sin(alpha)
        y_rot = x_shifted * fd.sin(alpha) + y_shifted * fd.cos(alpha)

        x_scaled = x_rot / a
        y_scaled = y_rot / b
        # Rotate global coordinates to align with airfoil
        

        # Apply elliptical scaling (stretch along x or y)
        

        elipse_circumference = np.pi*(3*(a+b)-np.sqrt((3*a+b)*(a+3*b)))

        # Compute the unrotated vortex velocity field
        u_x = -Gamma / elipse_circumference * y_scaled / (x_scaled**2 + y_scaled**2)
        u_y = Gamma / elipse_circumference * x_scaled / (x_scaled**2 + y_scaled**2)

        u_x = u_x * fd.cos(-alpha) - u_y * fd.sin(-alpha)
        u_y = u_x * fd.sin(-alpha) + u_y * fd.cos(-alpha)

        # Convert to firedrake vector function
        vortex.project(fd.as_vector([u_x, u_y]))

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
        converged = False
        while not converged:
            try:
                correction_model.solve(self.solver_params)
                self.solver_params["ksp_rtol"] = self.kwargs.get("min_rtol", 1e-14)
                converged = True
            except:
                self.solver_params["ksp_rtol"] *= 10
                print(f"Solver did not converge, increasing ksp_rtol to {self.solver_params['ksp_rtol']}")
        
        velocityPotential = correction_model.u_sol
        velocityPotential -= correction_model.u_sol.dat.data.min()

        # Computing the boundary correcting velocity field
        velocityBC = fd.Function(correction_model.W)
        velocityBC.project(fd.grad(velocityPotential))

        return velocityBC
    
    def __compute_pressure_coefficients(self, velocity, model):
        pressure = fd.Function(model.V, name = "Pressure_coeff")

        pressure.interpolate(1 - (fd.sqrt(fd.dot(velocity, velocity))/self.V_inf) ** 2)
        self.pressure_coeff = pressure
        if self.write:
            pressure_output = fd.VTKFile("pressure_output.pvd")
            pressure_output.write(self.pressure_coeff)
        return None



if __name__ == "__main__":

    task = input("""
    1: Compute lift coefficient for NACA airfoil
    2: Compute simple model
    Choose task: """)

    if task == "1":
        """Example for NACA 0012 airfoil
        NASA data from https://turbmodels.larc.nasa.gov/naca0012_val.html
           - Digitized Abbott and von Doenhoff CL data
        Data is in the form of [alpha, cl]
        NASA data is in degrees"""

        nasa_data = np.loadtxt("0012.abbottdata.cl.txt")
        eliptic_data = np.empty_like(nasa_data)
        eliptic_data[:,0] = nasa_data[:,0]
        circular_data = np.empty_like(nasa_data)
        circular_data[:,0] = nasa_data[:,0]
        kwargs_eliptic = {"ylim":[-4,4], "V_inf": 10, "g_div": 7.5, "write":False,
               "n_airfoil": 2000,
               "n_fs": 50,
               "n_bed": 50,
               "n_inlet": 20,
               "n_outlet": 20,
               "solver_params": {"ksp_type": "preonly", "pc_type": "lu", "ksp_rtol": 1e-14},}
        kwargs_circular = {"ylim":[-4,4], "V_inf": 10, "g_div": 5, "write":False,
               "n_airfoil": 2000,
               "n_fs": 50,
               "n_bed": 50,
               "n_inlet": 20,
               "n_outlet": 20,
               "a": 1,
               "b": 1,
               "solver_params": {"ksp_type": "preonly", "pc_type": "lu", "ksp_rtol": 1e-14},}

        for i,val in enumerate(nasa_data[:,0]):
            print(f"{i}/{len(nasa_data[:,0])}")
            print("Elliptic model")
            eliptic_model = PotentialFlowSolver("0012", 3, val, kwargs=kwargs_eliptic)
            eliptic_model.solve()
            eliptic_data[i,1] = eliptic_model.lift_coeff
            del(eliptic_model)

            print("Circular Model")
            circular_model = PotentialFlowSolver("0012", 3, val, kwargs=kwargs_circular)
            circular_model.solve()
            circular_data[i,1] = circular_model.lift_coeff
            del(circular_model)
            
        
        save_results = input("Save results? (y/n) \n This will overwrite the existing files")
        if save_results == "n":
            np.savetxt("../../Visualisation/PotentialFLowCL/circular_cl_point.txt",circular_data)
            np.savetxt("../../Visualisation/PotentialFLowCL/eliptic_cl_point.txt",eliptic_data)
    
    elif task == "2":
        kwargs = {"ylim":[-4,4], "V_inf": 10, "g_div": 1, "write":True,
               "n_airfoil": 1000,
               "n_fs": 30,
               "n_bed": 30,
               "n_inlet": 10,
               "n_outlet": 10,
               "g_div": 20, 
               "a": 1,
               "b": 1,
               "center_of_airfoil": (0,0),
               "min_tol": 1e-14,
               "solver_params": {"ksp_type": "preonly", "pc_type": "lu", "ksp_rtol": 1e-14}}

        model = PotentialFlowSolver("0012", P = 3, alpha = 20, kwargs=kwargs)
        model.solve()
        
