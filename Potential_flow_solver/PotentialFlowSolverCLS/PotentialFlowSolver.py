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
        # Fetching kwargs and setting standard parameters for the solver if they are not stated
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

        self.c0 = self.kwargs.get("g_div", 7)

        if "mesh" in self.kwargs:
            self.mesh = self.kwargs["mesh"]
        else:
            self.mesh = naca_mesh(self.airfoil, np.rad2deg(self.alpha), self.xlim, self.ylim, 
                              center_of_airfoil=self.center_of_airfoil,
                              n_airfoil = self.kwargs.get("n_airfoil"),
                              n_fs = self.kwargs.get("n_fs"),
                              n_bed = self.kwargs.get("n_bed"),
                              n_inlet = self.kwargs.get("n_inlet"),
                              n_outlet = self.kwargs.get("n_outlet"))

        if "fd_mesh" in self.kwargs:
            self.fd_mesh = self.kwargs["fd_mesh"]
        else:
            self.fd_mesh = meshio_to_fd(self.mesh)

        self.V = fd.FunctionSpace(self.fd_mesh, "CG", self.P)
        self.W = fd.VectorFunctionSpace(self.fd_mesh, "CG", self.P)

        self.a = self.kwargs.get("a", 1)
        self.b = self.kwargs.get("b", int(self.airfoil[2:])/100)

        self.solver_params = self.kwargs.get("solver_params", {"ksp_type": "preonly", "pc_type": "lu"})

        if "ksp_rtol" not in self.solver_params:
            self.solver_params["ksp_rtol"] = 1e-14

        self.VisualisationPath = "../../Visualisation/PotentialFlowCL/"
        # Handeling output files
        if self.write:
            if os.path.exists(self.VisualisationPath + "velocity_output"):
                shutil.rmtree(self.VisualisationPath + "velocity_output")
            if os.path.exists(self.VisualisationPath + "vortex_output"):
                shutil.rmtree(self.VisualisationPath + "vortex_output")
            if os.path.exists(self.VisualisationPath + "pressure_output"):
                shutil.rmtree(self.VisualisationPath + "pressure_output")

            try:
                os.remove(self.VisualisationPath + "velocity_output.pvd")
            except:
                pass
            try:
                os.remove(self.VisualisationPath + "vortex_output.pvd")
            except:
                pass
            try:
                os.remove(self.VisualisationPath + "pressure_output.pvd")
            except:
                pass
            
            self.velocity_output = fd.VTKFile(self.VisualisationPath + "velocity_output.pvd")
            self.vortex_output = fd.VTKFile(self.VisualisationPath + "vortex_output.pvd")
            self.BC_output = fd.VTKFile(self.VisualisationPath + "velocityBC.pvd")


    def solve(self):
        center_of_vortex = self.kwargs.get("center_of_vortex", self.center_of_airfoil)
        # Identify trailing edge and leading edge and use these coordinates to set a point a little distance from the trailing edge
        p1, p_te, p_leading_edge, pn= self.get_edge_info()
        vn = pn - p_te
        vn /= np.linalg.norm(vn)
        v1 = p1 - p_te
        v1 /= np.linalg.norm(v1)
        v12 = (vn - v1)
        p_te_new = p_te + np.array([v12[1], -v12[0]])/70


        # Initializing Laplaze solver
        if not self.kwargs.get("fs_DBC", np.array([0])).any():
            model = PoissonSolver(self.fd_mesh, P=self.P, nullspace=True)
        else:
            model = PoissonSolver(self.fd_mesh, P=self.P)
        v_in, v_out = self.__get_bcflux__()
        model.impose_NBC(fd.Constant(-v_in), self.kwargs.get("inlet", 1))
        model.impose_NBC(fd.Constant(v_out), self.kwargs.get("outlet", 2))

        # Free surface boundary condition
        if self.kwargs.get("fs_DBC", np.array([0])).any():
            xs = self.kwargs.get("fs_xs")
            ys = self.kwargs.get("fs_DBC")
            model.impose_DBC(interp1d(xs,ys), self.kwargs.get("fs", 4), "only_x")
        
        # Raise the tolorance for to see whether the solver can converge under a less strict tolorance
        converged = False
        try:
            while not converged:
                try:
                    model.solve(self.solver_params)
                    self.solver_params["ksp_rtol"] = self.kwargs.get("min_rtol", 1e-14)
                    converged = True
                except:
                    self.solver_params["ksp_rtol"] *= 10
                    print(f"Solver did not converge, increasing ksp_rtol to {self.solver_params['ksp_rtol']}")
                    if self.solver_params["ksp_rtol"] > 0.9:
                        model.solve(self.solver_params)
        except:
            raise BrokenPipeError("Potential flow solver initialization did not converge")

        # Standardizing the velocity potential to avoid overflow
        velocityPotential = model.u_sol
        velocityPotential -= np.min(velocityPotential.dat.data)

        # For exposing initial velocity potential for fs solver
        self.init_phi = velocityPotential.copy()

        # Computing the velocity field
        velocity = fd.Function(model.W, name="velocity")
        velocity.project(fd.as_vector((velocityPotential.dx(0), velocityPotential.dx(1))))
        vortex = fd.Function(model.W, name="vortex")
        
        # Prepare for writing the results in a ovd file
        if self.write:
            self.velocity_output.write(velocity)

        # Initializing main loop
        old_Gamma = 0
        vortex_sum = fd.Function(model.W, name="vortex")
        velocityBC_sum = fd.Function(model.W, name="Boundary correction")
        time_total = time()
        self.Gamma = 0
        Gammas = []
        
        # Main loop
        for it, _ in enumerate(range(self.kwargs.get("max_iter", 20) + 1)):
            # Start time
            time_it = time()

            # 
            if __name__ == "__main__" or self.kwargs.get("print_iter", False):
                print(f"Starting iteration {it}")
            else:
                print(f"Starting PotentialFLowSolver iteration {it}", end="\r")

            # Computing the vortex strength at the point a little out from the trailing edge
            vte = velocity.at(p_te_new)

            # Using the vortex strength a little out from the trailing estimate a new vortex strength
            Gamma = self.compute_vortex_strength(v12, vte, p_te_new)
            Gammas.append(Gamma)

            # Use an adaptive method to do feedback controlled scaling of the strength of the vortex
            Gamma = self.__compute_updated_Gamma__(Gammas)
            
            # Redifine Gamma as the scaled version
            Gammas[-1] = Gamma
            self.Gamma += Gamma

            # Compute the vortex
            vortex = self.compute_vortex(Gamma, model, center_of_vortex, vortex)

            # Sum velocity and the vortex to add circulation to the flow
            velocity += vortex
            vortex_sum += vortex

            # Computing the boundary correction
            velocityBC = self.compute_boundary_correction(vortex)
            velocity += velocityBC
            velocityBC_sum += velocityBC
            
            # Check for convergence and printing appropriate data
            if np.abs(np.dot(v12, vte)) < self.kwargs.get("dot_tol", 1e-6):
                print(f"PoissonSolver converged in {it} iterations")
                print(f"\t Total time: {time() - time_total}")
                print(f"\t dGamma: {np.abs(Gamma - old_Gamma)}")
                print(f"\t dot product: {np.dot(v12, vte)}")
                print(f"\n")
                break

            # Checking for Stagnation and printing appropriate data
            if np.abs(Gamma - old_Gamma) < self.kwargs.get("gamma_tol", 1e-6):
                print(f"PoissonSolver stagnated in {it-1} iterations")
                print(f"\t Total time: {time() - time_total}")
                print(f"\t dGamma: {np.abs(Gamma - old_Gamma)}")
                print(f"\t dot product: {np.dot(v12, vte)}")
                print(f"\n")
                break

            # Checking for divergence and printing appropriate data
            if np.abs(Gamma - old_Gamma) > 1e4:
                print(f"PoissonSolver diverged in {it} iterations")
                print(f"\t Total time: {time() - time_total}")
                print(f"\t dGamma: {np.abs(Gamma - old_Gamma)}")
                print(f"\t dot product: {np.dot(v12, vte)}")
                print(f"\n")
                break

            # Updating the vortex strength and printing appropriate data
            if __name__ == "__main__" or self.kwargs.get("print_iter", False):
                print(f"\t dGamma: {Gamma - old_Gamma}")
                print(f"\t dot product: {np.dot(velocity.at(p_te_new), v12)}")
                print(f"\t Iteration time: {time() - time_it} seconds\n")
            dgamma = np.abs(Gamma - old_Gamma)
            old_Gamma = Gamma

            # Write to file
            if self.write:
                self.velocity_output.write(velocity)
                self.vortex_output.write(vortex)
                vortex += velocityBC
                self.BC_output.write(vortex)

            # END OF MAIN LOOP

        # Compute lift based on circulation given the formula in the Kutta Jacowski theorem
        self.lift = -self.Gamma * self.V_inf * self.kwargs.get("rho", 1.225)
        self.lift_coeff = self.lift / (1/2 * self.kwargs.get("rho", 1.225) * self.V_inf**2)

        # Compute pressure coefficients
        self.__compute_pressure_coefficients(velocity, model)

        self.velocity = velocity

        # Print relevant information if solver did not converge
        if it == self.kwargs.get("max_iter", 20):
            print(f"PoissonSolver did not converge in {it} iterations")
            print(f"\t dGamma: {dgamma}")
            print(f"\t dot product: {np.dot(velocity.at(p_te_new), v12)}")
            print(f"\t Total time: {time() - time_total}")
            print(f"\n")
        

    def __get_bcflux__(self):
        # Find the length of the vectors that ensures that the ingoing and outgoind flux is v_inf * avg height of domain
        avg_height_of_domain = self.ylim[1] - self.ylim[0]
        # Defining the coords in the fd_mesh
        coords = fd.Function(self.W).interpolate(self.fd_mesh.coordinates).dat.data
        # For inlet
        boundary_indecies = self.V.boundary_nodes(self.kwargs.get("inlet", 1))
        boundary_coords = coords[boundary_indecies,:]
        v_in = avg_height_of_domain * self.V_inf / (np.max(boundary_coords[:,1]) - np.min(boundary_coords[:,1]))

        # For outlet
        boundary_indecies = self.V.boundary_nodes(self.kwargs.get("outlet", 2))
        boundary_coords = coords[boundary_indecies,:]
        v_out = avg_height_of_domain * self.V_inf / (np.max(boundary_coords[:,1]) - np.min(boundary_coords[:,1]))
        return v_in, v_out

    def __compute_updated_Gamma__(self, Gammas : list) -> float:
        # Adaptive stepsize controller for Gamma described in the report
        c0 = self.c0
        if len(Gammas) == 1:
            return Gammas[-1]/c0
        else:
            a = Gammas[-1]/c0/Gammas[-2]
            c1 = c0*(1-a)/(1-a**(len(Gammas)+1))
            self.c0 = c1
            return Gammas[-1]/c1

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
        rot_mat = np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
        p_leading_edge = rot_mat @ (-self.center_of_airfoil) + self.center_of_airfoil
        return p1, p_te, p_leading_edge, pn

    def compute_vortex_strength(self, v12, vte, p_te_new) -> float:
        """
        Computes the vortex strength for the given iteration
        """
        a = self.a
        b = self.b
        alpha = self.alpha

        # Get the coordinates of the trailing edge
        p_x = p_te_new[0]
        p_y = p_te_new[1]

        # Translating to center of airfoil
        x_t = p_x - self.center_of_airfoil[0]
        y_t = p_y - self.center_of_airfoil[1]

        # Rotating to align with airfoil
        x_bar = x_t * np.cos(alpha) - y_t * np.sin(alpha)
        y_bar = x_t * np.sin(alpha) + y_t * np.cos(alpha)

        # Computing vortex at trailing edge without Gamma/2pi
        Wx = -(y_bar/b) / (x_bar**2/a + y_bar**2/b)
        Wy = (x_bar/a) / (x_bar**2/a + y_bar**2/b)

        
        # Rotating back to global coordinates
        Wx_rot = Wx * np.cos(-alpha) - Wy * np.sin(-alpha)
        Wy_rot = Wx * np.sin(-alpha) + Wy * np.cos(-alpha)

        ellipse_circumference = np.pi*(3*(a+b) - np.sqrt(3*(a+b)**2+4*a*b))

        # Computing the vortex strength
        Gamma = -ellipse_circumference*(v12[0]*vte[0] + v12[1]*vte[1])/(Wx_rot*v12[0] + Wy_rot*v12[1])

        return Gamma

    def compute_vortex(self, Gamma, model, center_of_vortex, vortex) -> fd.Function:
        """
        Computes the vortex field for the given vortex strength
        """

        alpha = self.alpha  # Convert angle of attack to radians
        alpha = fd.Constant(alpha)

        # Extract airfoil scaling parameters
        a = fd.Constant(self.a)
        b = fd.Constant(self.b)
        
        # Translate coordinates
        x_translated = model.x - center_of_vortex[0]
        y_translated = model.y - center_of_vortex[1]
        
        # rotate the coordinates
        x_bar = (x_translated) * fd.cos(alpha) - (y_translated) * fd.sin(alpha)
        y_bar = (x_translated) * fd.sin(alpha) + (y_translated) * fd.cos(alpha)
        
        # Calculate the approximated circumference of the ellipse
        ellipse_circumference = fd.pi*(3*(a+b) - fd.sqrt(3*(a+b)**2+4*a*b))

        # Compute the unrotated elliptical vortex field
        u_x = -Gamma / ellipse_circumference * y_bar/b / ((x_bar/a)**2 + (y_bar/b)**2)
        u_y = Gamma / ellipse_circumference * x_bar/a / ((x_bar/a)**2 + (y_bar/b)**2)

        # Rotate the final vectors
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
        correction_model = PoissonSolver(self.fd_mesh, self.P, nullspace=True)

        # Imposing the Neumann boundary conditions
        correction_model.impose_NBC( -vortex, self.kwargs.get("inlet", 1))
        correction_model.impose_NBC( -vortex, self.kwargs.get("outlet", 2))
        correction_model.impose_NBC( -vortex, self.kwargs.get("bed", 3))
        correction_model.impose_NBC( -vortex, self.kwargs.get("fs", 4))
        correction_model.impose_NBC( -vortex, self.kwargs.get("naca", 5))


        # Solving the correction model and highering the tolorance if nessisary
        converged = False
        try:
            while not converged:
                try:
                    correction_model.solve(self.solver_params)
                    self.solver_params["ksp_rtol"] = self.kwargs.get("min_rtol", 1e-14)
                    converged = True
                except:
                    self.solver_params["ksp_rtol"] *= 10
                    print(f"Solver did not converge, increasing ksp_rtol to {self.solver_params['ksp_rtol']}")
                    if self.solver_params["ksp_rtol"] > 1e-5:
                        correction_model.solve(self.solver_params)
        except:
            raise BrokenPipeError("Boundary correction scheme did not converge")
        # standardize the result such that the minimum is 0 (this does nothing to the gradient and thus does nothing to the flow)
        velocityPotential = correction_model.u_sol
        velocityPotential -= np.min(velocityPotential.dat.data)

        # Computing the boundary correcting velocity field
        velocityBC = fd.Function(correction_model.W)
        velocityBC.project(fd.as_vector((velocityPotential.dx(0), velocityPotential.dx(1))))

        return velocityBC
    
    def __compute_pressure_coefficients(self, velocity, model):
        # Defining the firedrake function
        pressure = fd.Function(model.V, name = "Pressure_coeff")

        # Defining pressure coefficents in all of the domain from the formula given in the report.
        pressure.interpolate(1 - (fd.sqrt(fd.dot(velocity, velocity))/self.V_inf) ** 2)
        self.pressure_coeff = pressure

        # Write the results
        if self.write:
            pressure_output = fd.VTKFile(self.VisualisationPath + "pressure_output.pvd")
            pressure_output.write(self.pressure_coeff)
        return None



if __name__ == "__main__":

    task = input("""
    1: Compute lift coefficient for NACA airfoil
    2: Compute simple model
    3: n-Convergence and p-convergence of pressure coefficients around 0012 airfoil with eliptic and circular flow
    4: n-Convergence and p-convergence of lift coefficients around 0012 airfoil with eliptic and circular flow
    5: Compute information for pressure coefficient plot
    Choose task: """)

    if task == "1":
        """Example for NACA 0012 airfoil
        NASA data from https://turbmodels.larc.nasa.gov/naca0012_val.html
           - Digitized Abbott and von Doenhoff CL data
        Data is in the form of [alpha, cl]
        NASA data is in degrees"""

        # Fetch data from nasa
        nasa_data = np.loadtxt("0012.abbottdata.cl.txt")

        # Create arrays to hold data
        eliptic_data = np.empty_like(nasa_data)
        eliptic_data[:,0] = nasa_data[:,0]
        circular_data = np.empty_like(nasa_data)
        circular_data[:,0] = nasa_data[:,0]

        # Set parameters for solver using circular vortex and solver using elliptical vortex
        P = 2
        kwargs_eliptic = {"ylim":[-10,10], "V_inf": 10, "g_div": 100, "write":False,
               "n_airfoil": 800,
               "n_fs": 100,
               "n_bed": 100,
               "n_inlet": 50,
               "n_outlet": 50,
               "solver_params": {"ksp_type": "preonly", "pc_type": "lu", "ksp_rtol": 1e-14},}
        kwargs_circular = {"ylim":[-10,10], "V_inf": 10, "g_div": 100, "write":False,
               "n_airfoil": 800,
               "n_fs": 100,
               "n_bed": 100,
               "n_inlet": 50,
               "n_outlet": 50,
               "a": 1,
               "b": 1,
               "solver_params": {"ksp_type": "preonly", "pc_type": "lu", "ksp_rtol": 1e-14},}
        
        # solve models and note down the result of the lift
        for i,val in enumerate(nasa_data[:,0]):
            print(f"{i+1}/{len(nasa_data[:,0])}")
            print("Elliptic model")
            eliptic_model = PotentialFlowSolver("0012", P, val, kwargs=kwargs_eliptic)
            eliptic_model.solve()
            eliptic_data[i,1] = eliptic_model.lift_coeff
            del(eliptic_model)

            print("Circular Model")
            circular_model = PotentialFlowSolver("0012", P, val, kwargs=kwargs_circular)
            circular_model.solve()
            circular_data[i,1] = circular_model.lift_coeff
            del(circular_model)
            
        # Ask if user wants to save data
        while True:
            save_results = input("Save results? (y/n) \n This will overwrite the existing files: ").lower()
            if save_results not in ["y" , "y " , " y", "n", " n", "n "]:
                print(f"\t '{save_results}' is not a valid answer")
                continue
            else:
                break
        # Save data
        if save_results.lower() in ["y" , "y " , " y"]:
            print(os.getcwd())
            np.savetxt("../../Visualisation/LiftCoeffPlot/circular_cl_point.txt",circular_data)
            np.savetxt("../../Visualisation/LiftCoeffPlot/eliptic_cl_point.txt",eliptic_data)
    
    elif task == "2":
        # Defining a model and solving it
        kwargs = {"ylim":[-4,4], "V_inf": 10, "write":True,
           "n_airfoil": 750,
           "n_fs": 50,
           "n_bed": 50,
           "n_inlet": 20,
           "n_outlet": 20,
           "dot_tol": 1e-4, "a":1, "b":1}
        
        model = PotentialFlowSolver("0012", P = 2, alpha = 10, kwargs=kwargs)
        model.solve()
    
    elif task == "3":
        # Fetch relevant data for 
        nasa_9mil_10AoA = np.array([
            [.9483 ,  .1147],
            [.9000 ,  .0684],
            [.8503 ,  .0882],
            [.7998 ,  .0849],
            [.7497 ,  .0782],
            [.7003 ,  .0739],
            [.6502 ,  .0685],
            [.5997 ,  .0813],
            [.5506 ,  .0884],
            [.5000 ,  .0940],
            [.4503 ,  .1125],
            [.4000 ,  .1225],
            [.3507 ,  .1488],
            [.3002 ,  .1893],
            [.2501 ,  .2292],
            [.2004 ,  .2973],
            [.1504 ,  .3900],
            [.1000 ,  .5435],
            [.0755 ,  .6563],
            [.0510 ,  .8031],
            [.0251 , 1.0081],
            [.0122 , 1.0241],
            [0.    ,-2.6598],
            [.0135 ,-3.9314],
            [.0271 ,-3.1386],
            [.0515 ,-2.4889],
            [.0763 ,-2.0671],
            [.1012 ,-1.8066],
            [.1503 ,-1.4381],
            [.1994 ,-1.2297],
            [.2501 ,-1.0638],
            [.2999 , -.9300],
            [.3499 , -.8094],
            [.3994 , -.7131],
            [.4496 , -.6182],
            [.4997 , -.5374],
            [.5492 , -.4563],
            [.5994 , -.3921],
            [.6495 , -.3247],
            [.6996 , -.2636],
            [.7489 , -.1964],
            [.8003 , -.1318],
            [.8500 , -.0613],
            [.8993 , -.0021],
            [.9489 ,  .0795],
        ])
        # Define arrays that include the values for n_airfoil and which polynomial orders we want to use in our convergence 
        points_around_airfoil = np.array([20,40,50,70,100,150,200,300,400,600,800,1000,1400,1800,2400,3000])
        np.savetxt(f"../../Visualisation/P_Coeff_Convergence/n_airfoil.txt",points_around_airfoil)
        ps = np.array([1,2,3,4])
        np.savetxt(f"../../Visualisation/P_Coeff_Convergence/ps.txt",ps)

        # Do a loop changing value of p
        for P in ps:
            P = int(P)

            # Do a loop changing between circular and elliptical vortex
            for mode in ["c","e"]:

                # Prepare arrays to save data
                infnorm = np.zeros_like(points_around_airfoil, dtype=float)
                meannorm = np.zeros_like(points_around_airfoil, dtype=float)
                l2norm = np.zeros_like(points_around_airfoil, dtype=float)
                times = np.zeros_like(points_around_airfoil, dtype=float)

                # Do a loop for different values of n_airfoil
                for i,val in enumerate(points_around_airfoil):

                    # Define parameters for solver
                    kwargs = {"ylim":[-4,4], "V_inf": 10, "g_div": 70, "write":False,
                            "n_airfoil": val,
                            "n_fs": 40,
                            "n_bed": 40,
                            "n_inlet": 20,
                            "n_outlet": 20,
                            "dot_tol": 1e-4}
                    if mode == "c":
                        kwargs["a"] = 1
                        kwargs["b"] = 1
                    
                    # Initialize solver
                    model = PotentialFlowSolver("0012", alpha = 10.0228, P=P, kwargs = kwargs)

                    # Start time
                    it_time = time()

                    # Solve if possible, else put placeholders on the results
                    try:
                        model.solve()
                    except:
                        times[i] = np.nan
                        infnorm[i] = np.nan
                        meannorm[i] = np.nan
                        l2norm[i] = np.nan
                        continue

                    # End time
                    times[i] = time() - it_time

                    # Get information about airfoil position
                    p1, p_te, p_leading_edge, pn= model.get_edge_info()

                    # Construct the cord
                    cord = p_te-p_leading_edge

                    # Construct a vector orthorgonal to the cord
                    orthcord = np.array([-cord[1],cord[0]])
                    
                    # Map x values from the nasa data to the cord
                    xvals = nasa_9mil_10AoA[:,0]
                    all_coords = p_leading_edge + np.array([cord * i for i in xvals])

                    # Prepare array for data
                    pressure = np.zeros_like(xvals)

                    # Start a loop going through the different x values on the lower side of the airfoil
                    OnLower = True
                    for j in range(len(pressure)):

                        # Check whether the x value is at 0, and the points transition to the top halv of the airfoil in the data from nasa
                        if nasa_9mil_10AoA[j,0] == 0:
                            # Mark that the rest of the points is on the upper surface of the airfoil
                            OnLower = False
                            # Note the pressure at the leading edge
                            pressure[j] = model.pressure_coeff.at(p_leading_edge)
                            continue
                        if OnLower:
                            # Use cord and the vector orthogonal to the cord to map the x values from the data from nasa to a point on the lower airfoil surface
                            coords = all_coords[j,:]
                            coords -= 0.5 * orthcord * 0.12/0.2*(0.2969 * np.sqrt(coords[0]) - 0.1260 * coords[0] - 0.3516 * coords[0]**2 + 0.2843 * coords[0]**3 - 0.1036 * coords[0]**4)
                            while True:
                                try:
                                    # Start at the halfwaypoint towards the surface of the airfoil and walk towards the airfoil in small increments to make sure the pressure is taken exactly at the surface
                                    pressure[j] = model.pressure_coeff.at(coords)
                                    break
                                except:
                                    coords -= 1e-5 * orthcord
                        else:
                            # Use cord and the vector orthogonal to the cord to map the x values from the data from nasa to a point on the upper airfoil surface
                            coords = all_coords[j,:]
                            coords += 0.5 * orthcord * 0.12/0.2*(0.2969 * np.sqrt(coords[0]) - 0.1260 * coords[0] - 0.3516 * coords[0]**2 + 0.2843 * coords[0]**3 - 0.1036 * coords[0]**4)
                            while True:
                                try:
                                    # Start at the halfwaypoint towards the surface of the airfoil and walk towards the airfoil in small increments to make sure the pressure is taken exactly at the surface
                                    pressure[j] = model.pressure_coeff.at(coords)
                                    break
                                except:
                                    coords += 1e-5 * orthcord
                    
                    # Create the vector noting the absolute difference between the pressure in the data from nasa and the found pressure
                    abs_distance = abs(nasa_9mil_10AoA[:,1] - pressure)

                    # Save the different norms in their respective arrays
                    infnorm[i] = abs_distance.max()
                    meannorm[i] = abs_distance.mean()
                    l2norm[i] = np.linalg.norm(abs_distance)
                # Save the results
                results = np.vstack((infnorm, meannorm, l2norm, times), dtype=float)
                if mode == "e":
                    np.savetxt(f"../../Visualisation/P_Coeff_Convergence/P_Coeffs_errors_ellipse_P:{P}.txt",results)
                else:
                    np.savetxt(f"../../Visualisation/P_Coeff_Convergence/P_Coeffs_errors_circle_P:{P}.txt",results)
    
    elif task == "4":
        # Gather the data from nasa as a baseline
        Truths = np.array([
            [-12.2535, -1.25912],
            [-11.2222, -1.18135],
            [-10.1947, -1.06927],
            [-8.14138, -0.827958],
            [-6.25579, -0.638207],
            [-5.22822, -0.526128],
            [-4.19972, -0.422627],
            [-1.96944, -0.215533],
            [0., 0.],
            [0.940006, 0.120611],
            [1.96944, 0.215533],
            [2.99515, 0.34477],
            [3.85131, 0.439599],
            [4.87888, 0.551678],
            [5.90831, 0.6466],
            [7.96346, 0.870758],
            [10.1891, 1.12074],
            [11.0471, 1.19842]
        ])
        
        # Loop through the indexes that you are interested in seeing the convergence of
        for index in [10,17]:

            # Define arrays that include the values for n_airfoil and which polynomial orders we want to use in our convergence 
            points_around_airfoil = np.array([20,40,50,70,100,150,200,300,400,600,800,1000,1400,1800,2400,3000])
            np.savetxt(f"../../Visualisation/L_Coeff_Convergence/data/n_airfoil.txt",points_around_airfoil)
            ps = np.array([1,2,3,4])
            np.savetxt(f"../../Visualisation/L_Coeff_Convergence/data/ps.txt",ps)
            
            # Create a loop for the different values of p
            for P in ps:
                P = int(P)
                
                # Do a loop changing between circular and elliptical vortex
                for mode in ["c","e"]:

                    # Initialize arrays for saving error and time
                    error = np.zeros_like(points_around_airfoil, dtype=float)
                    times = np.zeros_like(points_around_airfoil, dtype=float)

                    # Do a loop for different values of n_airfoil
                    for i,val in enumerate(points_around_airfoil):
                        # Set solver settings
                        kwargs = {"ylim":[-4,4], "V_inf": 10, "g_div": 70, "write":False,
                                "n_airfoil": val,
                                "n_fs": 40,
                                "n_bed": 40,
                                "n_inlet": 20,
                                "n_outlet": 20,
                                "dot_tol": 1e-4}
                        if mode == "c":
                            kwargs["a"] = 1
                            kwargs["b"] = 1

                        # Initialize model
                        model = PotentialFlowSolver("0012", alpha = Truths[index,0], P=P, kwargs = kwargs)

                        # Start time
                        it_time = time()

                        # Solve if possible, else put placeholders on the results
                        try:
                            model.solve()
                        except:
                            times[i] = np.nan
                            error[i] = np.nan
                            continue

                        # Note the time the model took to solve, and the distance from the correct lift coefficient.
                        times[i] = time() - it_time
                        error[i] = (Truths[index,1] - model.lift_coeff)
                    results = np.vstack((error, times), dtype=float)

                    # Save the files
                    if mode == "e":
                        np.savetxt(f"../../Visualisation/L_Coeff_Convergence/data/L_Coeffs_errors_ellipse_P:{P}_AOA:{Truths[index,0]}.txt",results)
                    else:
                        np.savetxt(f"../../Visualisation/L_Coeff_Convergence/data/L_Coeffs_errors_circle_P:{P}_AOA:{Truths[index,0]}.txt",results)

    elif task == "5":
        # Fetch data from nasa
        nasa_9mil_10AoA = np.array([
            [.9483 ,  .1147],
            [.9000 ,  .0684],
            [.8503 ,  .0882],
            [.7998 ,  .0849],
            [.7497 ,  .0782],
            [.7003 ,  .0739],
            [.6502 ,  .0685],
            [.5997 ,  .0813],
            [.5506 ,  .0884],
            [.5000 ,  .0940],
            [.4503 ,  .1125],
            [.4000 ,  .1225],
            [.3507 ,  .1488],
            [.3002 ,  .1893],
            [.2501 ,  .2292],
            [.2004 ,  .2973],
            [.1504 ,  .3900],
            [.1000 ,  .5435],
            [.0755 ,  .6563],
            [.0510 ,  .8031],
            [.0251 , 1.0081],
            [.0122 , 1.0241],
            [0.    ,-2.6598],
            [.0135 ,-3.9314],
            [.0271 ,-3.1386],
            [.0515 ,-2.4889],
            [.0763 ,-2.0671],
            [.1012 ,-1.8066],
            [.1503 ,-1.4381],
            [.1994 ,-1.2297],
            [.2501 ,-1.0638],
            [.2999 , -.9300],
            [.3499 , -.8094],
            [.3994 , -.7131],
            [.4496 , -.6182],
            [.4997 , -.5374],
            [.5492 , -.4563],
            [.5994 , -.3921],
            [.6495 , -.3247],
            [.6996 , -.2636],
            [.7489 , -.1964],
            [.8003 , -.1318],
            [.8500 , -.0613],
            [.8993 , -.0021],
            [.9489 ,  .0795],
        ])
        
        # Gather values of n_airfoil, P and whether the vortex should be circular or elliptical
        n_airfoil = int(input("What should the value of n_airfoil be?: "))
        P = int(input("What should the value of P be?: "))
        circle_answer = input("Should the vortex be circular? [Y/n]: ").lower()
        if circle_answer == "y":
            circle = True
        elif circle_answer == "n":
            circle = False
        else:
            raise ValueError("Not a valid answer")
        
        # Set solver settings
        kwargs = {"ylim":[-10,10], "V_inf": 10, "g_div": 70, "write":False,
                "n_airfoil": n_airfoil,
                "n_fs": 100,
                "n_bed": 100,
                "n_inlet": 100,
                "n_outlet": 100,
                "dot_tol": 1e-4}
        if circle:
            kwargs["a"] = 1
            kwargs["b"] = 1

        # Initialize solver
        model = PotentialFlowSolver("0012", alpha = 10.0228, P=P, kwargs = kwargs)

        # Solve
        model.solve()

        # Get information about airfoil position
        p1, p_te, p_leading_edge, pn= model.get_edge_info()

        # Construct the cord
        cord = p_te-p_leading_edge

        # Construct a vector orthorgonal to the cord
        orthcord = np.array([-cord[1],cord[0]])
        
        # Map x values from the nasa data to the cord
        xvals = nasa_9mil_10AoA[:,0]
        all_coords = p_leading_edge + np.array([cord * i for i in xvals])

        # Prepare array for data
        pressure = np.zeros_like(xvals)

        # Start a loop going through the different x values on the lower side of the airfoil
        OnLower = True
        for j in range(len(pressure)):

            # Check whether the x value is at 0, and the points transition to the top halv of the airfoil in the data from nasa
            if nasa_9mil_10AoA[j,0] == 0:
                # Mark that the rest of the points is on the upper surface of the airfoil
                OnLower = False
                # Note the pressure at the leading edge
                pressure[j] = model.pressure_coeff.at(p_leading_edge)
                continue
            if OnLower:
                # Use cord and the vector orthogonal to the cord to map the x values from the data from nasa to a point on the lower airfoil surface
                coords = all_coords[j,:]
                coords -= 0.5 * orthcord * 0.12/0.2*(0.2969 * np.sqrt(coords[0]) - 0.1260 * coords[0] - 0.3516 * coords[0]**2 + 0.2843 * coords[0]**3 - 0.1036 * coords[0]**4)
                while True:
                    try:
                        # Start at the halfwaypoint towards the surface of the airfoil and walk towards the airfoil in small increments to make sure the pressure is taken exactly at the surface
                        pressure[j] = model.pressure_coeff.at(coords)
                        break
                    except:
                        coords -= 1e-5 * orthcord
            else:
                # Use cord and the vector orthogonal to the cord to map the x values from the data from nasa to a point on the upper airfoil surface
                coords = all_coords[j,:]
                coords += 0.5 * orthcord * 0.12/0.2*(0.2969 * np.sqrt(coords[0]) - 0.1260 * coords[0] - 0.3516 * coords[0]**2 + 0.2843 * coords[0]**3 - 0.1036 * coords[0]**4)
                while True:
                    try:
                        # Start at the halfwaypoint towards the surface of the airfoil and walk towards the airfoil in small increments to make sure the pressure is taken exactly at the surface
                        pressure[j] = model.pressure_coeff.at(coords)
                        break
                    except:
                        coords += 1e-5 * orthcord
        # Save results
        if circle:
            np.savetxt(f"../../Visualisation/P_Coeff_plot/data/P_Coeffs_pressure_circle_P:{P}_POA:{n_airfoil}.txt",pressure)
        else:
            np.savetxt(f"../../Visualisation/P_Coeff_plot/data/P_Coeffs_pressure_ellipse_P:{P}_POA:{n_airfoil}.txt",pressure)