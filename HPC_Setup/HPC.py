import numpy as np
import firedrake as fd

#==========================================================
#====================== MESHING ===========================
#==========================================================

import gmsh
import os
import meshio


def naca_4digit(string : str, n : int, alpha : float = 0, position_of_center : np.ndarray = np.array([0.5,0])) -> np.ndarray:
    """
    Returns the airfoil camber line and thickness distribution.
    Parameters
    ----------
    string : str
        A string consisting of 4 integers like "0012". This is the "name" of the airfoil
    n : int
        The number of points you want to modulate the airfoil with
    """

    # Raise error if name of NACA-airfoil is not 4 long
    if len(string) != 4:
        raise IndexError("The string needs to have the length 4")
    
    # Fetch information from the name of the airfoil
    m = int(string[0])/100
    p = int(string[1])/10 if string[1] != "0" else 0.1
    t = int(string[2] + string[3])/100

    # If there should be an equal amount of nodes on the airfoil
    if (n//2)*2 == n:

        # Create a linspace with points spaced equally between 0 and pi for the upper and lower surface of the airfoil
        beta = np.linspace(0,np.pi,(n//2)+1)

        # Convert this linspace to an array with points that are the most concentrated at the edges of the airfoil.
        x = (1-np.cos(beta))/2

        # Define the thickness of the airfoil distrebution in terms of x
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)

        # Define the camber in terms of x
        yc = np.where(x < p, m/p**2 * (2*p*x - x**2), m/(1-p)**2 * ((1-2*p) + 2*p*x - x**2))

        # Create the lower and upper part of the airfoil
        lower = np.hstack((x.reshape(-1,1), (yc - yt).reshape(-1,1)))[::-1][:-1]
        upper = np.hstack((x.reshape(-1,1), (yc + yt).reshape(-1,1)))[:-1]

    # If there should be an odd amount of nodes on the airfoil
    elif (n//2)*2 != n:

        # Create a linspace with points spaced equally between 0 and pi for the lower surface of the airfoil with one more point than for the upper part
        beta = np.linspace(0,np.pi,(n//2)+2)

        # Convert this linspace to an array with points that are the most concentrated at the edges of the airfoil.
        x = (1-np.cos(beta))/2
        
        # Define the thickness of the airfoil distrebution in terms of x
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)

        # Define the camber in terms of x
        yc = np.where(x < p, m/p**2 * (2*p*x - x**2), m/(1-p)**2 * ((1-2*p) + 2*p*x - x**2))

        # Define the lower part of the airfoil
        lower = np.hstack((x.reshape(-1,1), (yc - yt).reshape(-1,1)))[::-1][:-1]

        # Create a linspace with points spaced equally between 0 and pi for the upper surface of the airfoil
        beta = np.linspace(0,np.pi,(n//2)+1)

        # Convert this linspace to an array with points that are the most concentrated at the edges of the airfoil.
        x = (1-np.cos(beta))/2
        
        # Define the thickness of the airfoil distrebution in terms of x
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)

        # Define the camber in terms of x
        yc = np.where(x < p, m/p**2 * (2*p*x - x**2), m/(1-p)**2 * ((1-2*p) + 2*p*x - x**2))

        # Define the upper part of the airfoil
        upper = np.hstack((x.reshape(-1,1), (yc + yt).reshape(-1,1)))[:-1]
    
    # Stack the points of the lower surface on top of the points from the upper surface
    points = np.vstack((lower, upper))

    # Relocate the airfoil such that it haves its center at the origin
    points -= np.array([0.5,0])

    # Rotate the airfoil
    alpha = np.deg2rad(alpha)
    rot_matrix = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    points = np.dot(points, rot_matrix)

    # Relocate the airfoil such that it haves its center as desired
    points += position_of_center

    # Return points defining the airfoil
    return points

def naca_mesh(airfoil: str, alpha: float = 0, xlim: tuple = (-7,13), ylim: tuple = (-2,1), **kwargs):
    """
    
    parameters
    ----------
    airfoil: str
        directory of the airfoil coordinates in txt format
    alpha: float
        angle of attack in degrees
    xlim: tuple
        x limits of the domain
    ylim: tuple
        y limits of the domain
    
    ** kwargs:
        - n_in, n_out, n_bed, n_fs, n_airfoil: int
            Number of points in the inlet, outlet, bed, front step and airfoil
        - prog_in, prog_out, prog_bed, prog_fs: float
            Progression in the inlet, outlet, bed and front step
        - scale: float
            Element scale factor
        
    """

    # ==================== Initializing the model ====================
    try:
        kwargs = kwargs["kwargs"]
    except:
        pass
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)

    # Creating domain
    xmin, xmax = xlim
    ymin, ymax = ylim

    scale = kwargs.get('scale', 1.034)
    gmsh.model.geo.addPoint(xmin, ymin, 0, scale, tag=1) # bottom left
    gmsh.model.geo.addPoint(xmin, ymax, 0, scale, tag=2) # top left
    gmsh.model.geo.addPoint(xmax, ymin, 0, scale, tag=3) # bottom right
    gmsh.model.geo.addPoint(xmax, ymax, 0, scale, tag=4) # top right

    inlet = gmsh.model.geo.addLine(1, 2, tag=1)
    outlet = gmsh.model.geo.addLine(3, 4, tag=2)
    bed = gmsh.model.geo.addLine(1, 3, tag=3)
    fs = gmsh.model.geo.addLine(2, 4, tag=4)

    boundary_loop = gmsh.model.geo.addCurveLoop([inlet, fs, -outlet, -bed], tag=1)

    # ==================== Handling the airfoil ====================
    n_airfoil = kwargs.get('n_airfoil') if kwargs.get('n_airfoil') else 60
    coords = naca_4digit(airfoil, n = n_airfoil, alpha=alpha, position_of_center=kwargs.get("center_of_airfoil",np.array([0.5,0])))


    points = []
    for idx, coord in enumerate(coords[::len(coords)//n_airfoil]):
        points.append(gmsh.model.geo.addPoint(coord[0], coord[1], 0, scale, tag=5+idx))
    
    # Create lines
    lines = []
    for idx in range(len(points)-1):
        line = gmsh.model.geo.addLine(points[idx], points[idx+1], tag = 4+idx + 1)
        lines.append(line)
    
    
    line = gmsh.model.geo.addLine(points[-1], points[0], tag = 4+len(points))
    lines.append(line)

    airfoil_line = gmsh.model.geo.addCurveLoop(lines, tag=5)

    # ==================== Creating the surface ====================

    # Create the surface
    gmsh.model.geo.addPlaneSurface([boundary_loop, airfoil_line], tag=1)

    gmsh.model.geo.synchronize()

    # ==================== Physical groups ====================

    # Inlet
    gmsh.model.addPhysicalGroup(1, [inlet], tag=1)
    gmsh.model.setPhysicalName(1, 1, 'inlet')

    # Outlet
    gmsh.model.addPhysicalGroup(1, [outlet], tag=2)
    gmsh.model.setPhysicalName(1, 2, 'outlet')

    # Bed
    gmsh.model.addPhysicalGroup(1, [bed], tag=3)
    gmsh.model.setPhysicalName(1, 3, 'bed')

    # Free surface
    gmsh.model.addPhysicalGroup(1, [fs], tag=4)
    gmsh.model.setPhysicalName(1, 4, 'free_surface')

    # Airfoil
    gmsh.model.addPhysicalGroup(1, lines, tag=5)
    gmsh.model.setPhysicalName(1, 5, 'airfoil')

    # Domain
    gmsh.model.addPhysicalGroup(2, [1], tag=6)
    gmsh.model.setPhysicalName(2, 6, 'domain')
    # ==================== Transfinte curves ====================
    # Inlet
    n_in = kwargs.get('n_in') if kwargs.get('n_in') else 70
    prog_in = kwargs.get('prog_in', 0.99)
    gmsh.model.mesh.setTransfiniteCurve(tag = inlet, numNodes=n_in, coef=prog_in)

    # Outlet
    n_out = kwargs.get('n_out') if kwargs.get('n_out') else 70
    prog_out = kwargs.get('prog_out', 0.99)
    gmsh.model.mesh.setTransfiniteCurve(tag = outlet, numNodes=n_out, coef=prog_out)

    # Bed
    n_bed = kwargs.get('n_bed') if kwargs.get('n_bed') else 200
    prog_bed = kwargs.get('prog_bed', 1)
    gmsh.model.mesh.setTransfiniteCurve(tag = bed, numNodes=n_bed, coef=prog_bed)

    # Free surface
    n_fs = kwargs.get('n_fs') if kwargs.get('n_fs') else 200
    prog_fs = kwargs.get('prog_fs', 1)
    gmsh.model.mesh.setTransfiniteCurve(tag = fs, numNodes=n_fs, coef=prog_fs)

    # Airfoil
    prog_airfoil = kwargs.get('prog_airfoil', 1)
    for line in lines:
        gmsh.model.mesh.setTransfiniteCurve(tag = line, numNodes=n_airfoil//len(lines), coef=prog_airfoil)

    # ==================== Meshing and writing model ====================
    gmsh.model.mesh.generate(2)

    if kwargs.get('test', False):
        gmsh.fltk.run()

    if kwargs.get("write", False):
        out_name = kwargs.get('o', "naca")
        file_type = kwargs.get("file_type", "msh")
        gmsh.write(out_name + "." + file_type)

    # ==================== Converting to meshio ====================
    gmsh.write("temp.msh")

    mesh = meshio.read("temp.msh")
    os.system("rm temp.msh")
    
    gmsh.finalize()

    return mesh

def meshio_to_fd(mesh: meshio.Mesh):
    """
    Converts a meshio mesh to a firedrake mesh
    """

    # Write a temperary gmsh file from meshio mesh
    meshio.write("temp.msh", mesh, file_format="gmsh22")

    # Read that gmsh file as a firedrake mesh
    fd_mesh = fd.Mesh("temp.msh")

    # Delete the temperary gmsh file
    os.system("rm temp.msh")

    # Return the firedrake mesh
    return fd_mesh

def shift_surface(mesh : fd.Mesh, func_b : callable, func_a : callable) -> meshio.Mesh:
    '''
    params
    ---
    func_b: callable 
        - Function before
    func_a: callable
        - Function after
    '''
    # Define p=1 functionspace
    V1 = fd.FunctionSpace(mesh, "CG", 1)

    coords = mesh.coordinates.dat.data
    
    # Find airfoil indecies
    naca_idx = V1.boundary_nodes(5)

    # Define maximal y value of coordinates on airfoil 
    M = np.max(coords[naca_idx][:,1])

    # Mask all points above the largest value of the airfoil
    point_mask = np.where((coords[:,1] > M))

    # Manipulate functions
    func_before = lambda x: func_b(x)-M
    func_after = lambda x: func_a(x)-M
    func_before = np.vectorize(func_before)
    func_after = np.vectorize(func_after)
    
    # Manipulate point values as stated in the report under methodology for free surface.
    coords[point_mask,1] -= M
    coords[point_mask,1] *= (func_after(coords[point_mask,0]) /  func_before(coords[point_mask,0]))
    coords[point_mask,1] += M

    # Set new coordinates of mesh
    mesh.coordinates.dat.data[:] = coords

    # Return mesh
    return mesh


#==========================================================
#=================== POISSON SOLVER =======================
#==========================================================

import matplotlib.pyplot as plt

class PoissonSolver:

    ###### Attributes ######
    # Setup
    mesh: fd.Mesh
    P: int

    V: fd.FunctionSpace
    W: fd.VectorFunctionSpace

    u: fd.Function
    v: fd.TestFunction

    # Problem
    a: fd.Form
    L: fd.Form
    f: fd.Function

    # BCs
    DirBCs: list
    x: fd.SpatialCoordinate
    y: fd.SpatialCoordinate

    # Solution
    u_sol: fd.Function
    true_sol: fd.Function


    ########## SETUP METHODS ##########
    def __init__(self, mesh: fd.Mesh, P = 1, nullspace=False):
        """Initializing solver befor BC are given"""
        self.mesh = mesh
        self.P = P
        self.has_nullspace = nullspace

        self.V = fd.FunctionSpace(self.mesh, "CG", self.P)
        self.W = fd.VectorFunctionSpace(self.mesh, "CG", self.P)
        self.u = fd.TrialFunction(self.V)
        self.v = fd.TestFunction(self.V)


        self.f = fd.Function(self.V)
        self.f = fd.Constant(0.0)

        self.a = fd.inner(fd.grad(self.u), fd.grad(self.v)) * fd.dx
        self.L = self.f * self.v * fd.dx


        self.DirBCs = []
        self.x, self.y = fd.SpatialCoordinate(self.mesh)

        self.u_sol = fd.Function(self.V)
        self.true_sol = None

        fs_indecies = self.V.boundary_nodes(4)
        self.fs_points = (fd.Function(self.W).interpolate(self.mesh.coordinates).dat.data)[fs_indecies,:]
        self.fs_xs = self.fs_points[:,0]
    
    def impose_rhs(self, rhs_func: fd.Function, func_type = "fd"):
        """Impose the right-hand side of the Poisson problem

        Args:
            rhs_func: callable
                Function that represents the right-hand side of the Poisson 
            
            func_type: str
                Type of the right-hand side function
        """
        if func_type == "fd":
            self.f = rhs_func
        elif func_type == "callable":
            self.f = fd.Function(self.V)
            self.f.interpolate(rhs_func(self.x, self.y))
        else:
            raise ValueError("Right-hand side must be a firedrake.function.Function or a callable object")
        
        self.L = (-self.f) * self.v * fd.dx
    
    def MMS(self, true_sol_func: fd.Function, DBCs: list[int] = [], NBCs: list[int] = [], func_type = "fd"):
        """ Method of manufactured solutions

        Args:
            true_sol_func: callable
                Function that represents the true solution of the Poisson problem
            
            DBCs: list[int]
                List of indices/tags of the Dirichlet boundary conditions
            
            NBCs: list[int]
                List of indices/tags of the Neumann boundary conditions
        """
        if func_type == "fd":
            self.true_sol = true_sol_func
        elif func_type == "callable":
            self.true_sol = fd.Function(self.V)
            self.true_sol.interpolate(true_sol_func(self.x, self.y))
        else:
            raise ValueError("True solution must be a firedrake.function.Function or a callable object")


        self.f = -fd.div(fd.grad(self.true_sol))
        self.L = self.f * self.v * fd.dx

        for DBC in DBCs:
            self.impose_DBC(self.true_sol, DBC)
        # 
        # for NBC in NBCs:
            # self.impose_NBC(fd.grad(self.true_sol), NBC)



    ########## BOUNDARY METHODS ##########
    def impose_DBC(self, bc_func: callable, bc_idx: int|list[int], func_type = "fd", target_point = np.array([0,0])):
        """Impose Dirichlet boundary conditions
        
        Args:
            bc_func: callable
                Function that represents the boundary condition
            bc_idx: list[int] | int
                Index/tag of the boundary
        """

        if func_type == "fd":
            self.DirBCs.append(fd.DirichletBC(self.V, bc_func, bc_idx))

        elif func_type == "callable":
            bc = fd.Function(self.V)
            bc.interpolate(bc_func(self.x, self.y))
            self.DirBCs.append(fd.DirichletBC(self.V, bc,bc_idx))
        
        elif func_type == "only_x":
            bc = fd.Function(self.V)
            bc.interpolate(bc_idx)
            coords = fd.Function(self.W).interpolate(self.mesh.coordinates).dat.data
            
            # find boundary coords
            boundary_indecies = self.V.boundary_nodes(bc_idx)
            boundary_coords = coords[boundary_indecies,:]
            
            # Interpolate using x-coordinate
            x_vals = boundary_coords[:, 0]
            bc.dat.data[boundary_indecies] = bc_func(x_vals)

            self.DirBCs.append(fd.DirichletBC(self.V, bc, bc_idx))
        
        elif func_type == "single_point":
            bc = fd.Function(self.V)
            bc.interpolate(bc_func(self.x, self.y))
            coords = fd.Function(self.W).interpolate(self.mesh.coordinates).dat.data
            idx = np.argmin(np.linalg.norm(coords - target_point, axis=1))
            # Create DirichletBC by passing a list of dof indices
            idx = int(idx)  # convert from numpy.int64
            bc_point = fd.DirichletBC(self.V, bc, [idx])
            self.DirBCs.append(bc_point)

        
        else:
            raise ValueError("Boundary condition must be a firedrake.function.Function or a callable object")

    
    def impose_NBC(self, bc_func: fd.Function, bc_idx: list[int], func_type = "fd"):
        """Impose Neumann boundary conditions
        
        Args:
            bc: fd.Function
                Function that represents the boundary condition
            bc_idx: int
                Index/tag of the boundary
        """
        # Ensure looping over the indices are possible for integer input
        bc_idx = [bc_idx] if type(bc_idx) == int else bc_idx

        if func_type == "fd":
            # Handeling scalar Neumann BCs
            if bc_func.ufl_shape == ():
                for idx in bc_idx:
                    self.L += bc_func * self.v * fd.ds(idx)

            # Handeling vector Neumann BCs
            elif bc_func.ufl_shape != (): # change to == (2,) for 2D only
                n = fd.FacetNormal(self.mesh)
                for idx in bc_idx:
                    self.L += fd.inner(bc_func, n) * self.v * fd.ds(idx)
        
        elif func_type == "callable":
            # Scalar Neumann BCs
            bc = fd.Function(self.V)
            bc.interpolate(bc_func(self.x, self.y))
            for idx in bc_idx:
                self.L += bc * self.v * fd.ds(idx)

        

        
    
    ########## SOLUTION AND PLOTTING METHODS ##########
    def solve(self, solver_params: dict = {"ksp_type": "cg"}):
        """Solve the Poisson problem"""
        if self.has_nullspace:
            nullspace = fd.VectorSpaceBasis(constant=True, comm=self.V.mesh().comm)  # nullspace = span{1}
            # (Optional but recommended)
            nullspace.orthonormalize()
            fd.solve(self.a == self.L, self.u_sol, bcs=self.DirBCs, solver_parameters=solver_params, nullspace=nullspace)
        else:
            fd.solve(self.a == self.L, self.u_sol, bcs=self.DirBCs, solver_parameters=solver_params)
        # except:
        #     # Create a function in the same space as the solution
        #     const_mode = fd.Function(self.V)
        #     const_mode.assign(1.0)  # Set all values to 1 (constant null space)

        #     # Define the null space correctly
        #     nullspace = fd.VectorSpaceBasis([const_mode])
        #     nullspace.orthonormalize()
        #     # A = fd.assemble(self.a)
        #     # A.setNullSpace(nullspace)
        #     fd.solve(self.a == self.L, self.u_sol, bcs=self.DirBCs, nullspace=nullspace, solver_parameters=solver_params)


#==========================================================
#=================== POTENTIALFLOW SOLVER =================
#==========================================================
from time import time
import shutil
from scipy.integrate import interp1d

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

        self.VisualisationPath = "./HPC_RESULTS/PotentialFlow/"
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
            if self.kwargs.get("print_iter", False):
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


#==========================================================
#=================== FREE SURFACE SOLVER =================
#==========================================================


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
        self.save_results = self.kwargs.get("save_results", True)
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

        self.visualisationpath = "../HPC_RESULTS/paraview"

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

        self.etas = np.zeros((self.kwargs.get("max_iter_fs", 10), len(self.fs_xs)), dtype=np.float32)
        self.phis = np.zeros((self.kwargs.get("max_iter_fs", 10), len(self.fs_xs)), dtype=np.float32)
        self.fs_xs_array = np.zeros((self.kwargs.get("max_iter_fs", 10), len(self.fs_xs)))
        self.residual_array = np.zeros((self.kwargs.get("max_iter_fs", 10), 2))
        
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
            
            # Update eta and dirichlet boundary condition and notify if the dolve does not converge
            try:
                new_eta, self.PhiTilde, residuals = self.__compute_fs_equations_weak1d__()
            except:
                raise BrokenPipeError("Free surface equations did not konverge")
            kwargs_for_Kutta_kondition["fs_DBC"] = self.PhiTilde
            kwargs_for_Kutta_kondition["fs_xs"] = self.fs_xs

            # Update mesh with new eta
            self.__update_mesh_data__(old_eta, new_eta)
            kwargs_for_Kutta_kondition["mesh"] = self.mesh
            kwargs_for_Kutta_kondition["fd_mesh"] = self.fd_mesh

            # Save result data while Loop is running
            if self.save_results:
                self.__save_results__(new_eta, residuals, i)

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
        #bcs_phi_in = fd.DirichletBC(W.sub(1), 0.0, 1) # Dirichlet BC for phi

        bcs = [bcs_eta]

        a1 = fd.inner(eta_n1 - eta_n, v_1)*fd.dx + fd.inner(eta_damp_in, v_1)*fd.dx + fd.inner(eta_damp_out, v_1)*fd.dx
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
    
    def __save_results__(self, new_eta, residuals, iter):
        self.etas[iter, :] = new_eta.copy()
        self.phis[iter, :] = self.PhiTilde.copy()
        self.fs_xs_array[iter, :] = self.fs_xs.copy()
        self.residual_array[iter] = np.array([residuals.copy(), self.dt*(iter+1)])
        np.save("./HPC_RESULTS/arrays/eta.npy", self.etas)
        np.save("./HPC_RESULTS/arrays/phiTilde.npy", self.phis)
        np.save("./HPC_RESULTS/arrays/fs_xs.npy", self.fs_xs_array)
        np.save("./HPC_RESULTS/arrays/residuals.npy", self.residual_array)

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
    kwargs = {"ylim":[-4,1], "xlim":[-8,27], 
            "xd_in": -6, "xd_out": 25,

            "write":True, "save_results": True,
            "V_inf": 1, 
            "g_div": 7, 
            "write":True,
            "n_airfoil": 501,
            "n_fs": 350,
            "n_bed": 120,
            "n_in": 30,
            "n_out": 30,
            "rtol": 1e-8,
            "a":1, "b":1,
            "max_iter": 50,
            "dot_tol": 1e-4,

            "fs_rtol": 1e-7,
            "max_iter_fs":5000,
            
            "dt": 5e-3,
            "damp":50}
    
    FS = FsSolver("0012", alpha = 5, P=2, kwargs = kwargs)
    FS.solve()

