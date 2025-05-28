print("Fetching libraries...", end = "\r")
import firedrake as fd
from firedrake.__future__ import interpolate
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.getcwd())


from Meshing.mesh_library import *
print("Libraries fetched!\n")



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
    def __init__(self, mesh: fd.Mesh, P = 1):
        """Initializing solver befor BC are given"""
        self.mesh = mesh
        self.P = P

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
        
        if __name__ == "__main__":
            print("PoissonSolver initialized!")
    
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
    def impose_DBC(self, bc_func: callable, bc_idx: int|list[int], func_type = "fd"):
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

    def plot_results(self, levels:int = 50, norm: str = "H1", xlim = None, ylim = None, figsize = None):
        """Plot the solution"""
        if self.true_sol is not None:
            fig, axes = plt.subplots(1, 3, figsize = (15, 5))
            p1 = fd.tricontourf(self.u_sol, axes=axes[0], levels = levels)
            axes[0].set_title("Numerical solution")

            p2 = fd.tricontourf(self.true_sol, axes=axes[1], levels = levels)
            axes[1].set_title("True solution")


            diff = fd.Function(self.V)
            diff.interpolate(self.u_sol - self.true_sol)
            p3 = fd.tricontourf(diff, axes=axes[2], levels = levels)
            axes[2].set_title(f"Error, $E_{{{norm}}}=$ {np.round(fd.errornorm(self.true_sol, self.u_sol, norm),3)}")

            if xlim is not None:
                for ax in axes:
                    ax.set_xlim(xlim)
            if ylim is not None:
                for ax in axes:
                    ax.set_ylim(ylim)

            fig.colorbar(p1, ax=axes[0])
            fig.colorbar(p2, ax=axes[1])
            fig.colorbar(p3, ax=axes[2])


            plt.show()

        else:
            figsize = (10,5) if figsize is None else figsize
            fig, ax = plt.subplots(1, 1, figsize = figsize)

            p1 = fd.tricontourf(self.u_sol, levels = levels, axes = ax)
            fig.colorbar(p1, ax=ax)

            ax.set_title("Nummerical solution")

            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
            
            plt.show()
        

if __name__ == "__main__":

    task = input("""
    1: Generate h and p Convergence data
    2: Generate h/p convergence data
    3: Simple test
    
    Choose task: """)

    if task == "1":
        from time import time
        print("Computing h-convergence...\n")

        # Meshes resolutions (3 time h = 4 to reset cache for accurate time measurements)
        hs = [4, 4, 4, 5, 10, 100, 200, 300, 500, 1000]#np.array([10, 50, 100, 200, 300, 400, 500])
        error_h = []

        true_sol = lambda x,y: fd.sin(x)*fd.sin(y)
        rhs = lambda x,y: -2*fd.sin(x)*fd.sin(y)
        NBC1 = lambda x,y: -fd.cos(x)*fd.sin(y)
        NBC2 = lambda x,y: fd.cos(x)*fd.sin(y)

        for h in hs:
            print(f"Computing for h = {h}...")
            t1 = time()
            # Mesh
            mesh = fd.UnitSquareMesh(h, h)

            # Solver
            model = PoissonSolver(mesh, P = 1)

            # Imposing true solution
            model.MMS(true_sol, DBCs=[3, 4], func_type="callable")
            model.impose_rhs(rhs, func_type="callable")
            model.impose_NBC(NBC1, 1, func_type="callable")
            model.impose_NBC(NBC2, 2, func_type="callable")

            # Solve
            model.solve()

            time_taken = time() - t1

            if h != 4:
                print(f"\t\t Time elapsed: {time_taken:.2f} s")
                err = fd.errornorm(model.true_sol, model.u_sol, norm_type="L2")
                print(f"\t\t Error: {err}\n")
                # Compute error
                error_h.append(np.array([h, err, time_taken]))

        print(error_h, "\n")   

        print("Computing p-convergence...\n")

        # Setup
        Ps = np.arange(1, 10)
        error_p = []

        for P in Ps:
            print(f"\t Computing for P = {P}...")
            t1 = time()

            # Mesh
            mesh = fd.UnitSquareMesh(5, 5)

            # Solver
            model = PoissonSolver(mesh, P = int(P))

            # Imposing true solution
            model.MMS(true_sol, DBCs=[3, 4], func_type="callable")
            model.impose_rhs(rhs, func_type="callable")
            model.impose_NBC(NBC1, 1, func_type="callable")
            model.impose_NBC(NBC2, 2, func_type="callable")

            # Solve
            model.solve()

            time_taken = time() - t1
            err = fd.errornorm(model.true_sol, model.u_sol, norm_type="L2")
            print(f"\t\t Time elapsed: {time_taken:.2f} s")
            print(f"\t\t Error: {err} \n")

            # Compute error
            error_p.append(np.array([P, err, time_taken]))
        print(error_p, "\n")

        
        error_h = np.array(error_h)
        error_p = np.array(error_p)

        save_data = input("Save new results? (y/n): ")
        if save_data == "y":
            np.savetxt("./PoissonSolver/PoissonCLS/PoissonError_h.txt", error_h)
            np.savetxt("./PoissonSolver/PoissonCLS/PoissonError_p.txt", error_p)
        else:
            print("Results not saved")

    elif task == "2":
        ps = np.array([1,2,3])
        hs = np.array([5, 10, 50, 100, 150])

        error_hp = []

        true_sol = lambda x,y: fd.sin(x)*fd.sin(y)
        rhs = lambda x,y: -2*fd.sin(x)*fd.sin(y)
        NBC1 = lambda x,y: -fd.cos(x)*fd.sin(y)
        NBC2 = lambda x,y: fd.cos(x)*fd.sin(y)

        for p in ps:
            for h in hs:
                print(f"Computing for P = {p} and h = {h}...")
                # Mesh
                mesh = fd.UnitSquareMesh(int(h/p), int(h/p))

                # Solver
                model = PoissonSolver(mesh, P = int(p))

                # Imposing true solution
                model.MMS(true_sol, DBCs=[3, 4], func_type="callable")
                model.impose_rhs(rhs, func_type="callable")
                model.impose_NBC(NBC1, 1, func_type="callable")
                model.impose_NBC(NBC2, 2, func_type="callable")

                # Solve
                model.solve()

                err = fd.errornorm(model.true_sol, model.u_sol, norm_type="L2")
                print(f"\t\t Error: {err} \n")

                # Compute error
                error_hp.append(np.array([p, model.V.dof_count, err]))
        
        # Save data
        save_data = input("Save new results? (y/n): ")
        if save_data == "y":
            np.savetxt("./PoissonSolver/PoissonCLS/PoissonError_hp.txt", error_hp)
        else:
            print("Results not saved")

    elif task == "3":
        mesh_kwargs = {
            "n_in": 20,
            "n_out": 20,
            "n_bed": 50,
            "n_fs": 50, 
            "n_airfoil" : 100
        }
        mesh = naca_mesh("0012", alpha = 15, ylim = (-4, 4), kwargs = mesh_kwargs)
        fd_mesh = meshio_to_fd(mesh)

        V_inf = fd.Constant(10)
        model = PoissonSolver(fd_mesh, P = 1)

        model.impose_NBC(-V_inf, 1, func_type="fd")
        model.impose_NBC(V_inf, 2, func_type="fd")

        model.solve(solver_params={"ksp_type": "cg", "pc_type": "sor"})
        outfile = fd.VTKFile("./PoissonSolver/PoissonCLS/Poisson_solution.pvd")
        outfile.write(model.u_sol)

        
        
    