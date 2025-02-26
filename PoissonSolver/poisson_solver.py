print("Fetching libraries...", end = "\r")
import firedrake as fd
from firedrake.__future__ import interpolate
import numpy as np
import matplotlib.pyplot as plt
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


        self.bcs = []
        self.x, self.y = fd.SpatialCoordinate(self.mesh)

        self.u_sol = fd.Function(self.V)
        true_sol = None

        
        if __name__ != "__main__":
            print("PoissonSolver initialized!")
    
    def impose_rhs(self, rhs_func: fd.Function):
        """Impose the right-hand side of the Poisson problem

        Args:
            rhs_func: callable
                Function that represents the right-hand side of the Poisson problem
        """
        self.f.assign(interpolate(rhs_func, self.V))
        self.L = self.f * self.v * fd.dx
    
    def impose_true_sol(self, true_sol_func: fd.Function):
        """Impose the true solution of the Poisson problem for method of manufactured solutions, 
        and automatically sets the right-hand side

        Args:
            true_sol_func: callable
                Function that represents the true solution of the Poisson problem
        """
        self.true_sol = true_sol_func

        self.f = -fd.div(fd.grad(self.true_sol))
        self.L = self.f * self.v * fd.dx



    ########## BOUNDARY METHODS ##########
    def impose_DBC(self, bc_func: fd.Function, bc_idx: int):
        """Impose Dirichlet boundary conditions
        
        Args:
            bc_func: callable
                Function that represents the boundary condition
            bc_idx: int
                Index/tag of the boundary
        """
        self.bcs.append(fd.DirichletBC(self.V, bc_func, bc_idx))
    
    def impose_NBC(self, bc_func: fd.Function, bc_idx: int):
        """Impose Neumann boundary conditions
        
        Args:
            bc: fd.Function
                Function that represents the boundary condition
            bc_idx: int
                Index/tag of the boundary
        """

        if bc_func.ufl_shape == ():
            self.L += bc_func * self.v * fd.ds(bc_idx)

        elif bc_func.ufl_shape == (2,):
            n = fd.FacetNormal(self.mesh)
            self.L += fd.inner(bc_func, n) * self.v * fd.ds(bc_idx)
        
    
    ########## SOLUTION AND PLOTTING METHODS ##########
    def solve(self, solver_params: dict = {"ksp_type": "cg"}):
        """Solve the Poisson problem"""
        fd.solve(self.a == self.L, self.u_sol, bcs=self.bcs, solver_parameters=solver_params)

    def plot_results(self, levels:int = 50, norm: str = "H1"):
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

            fig.colorbar(p1, ax=axes[0])
            fig.colorbar(p2, ax=axes[1])
            fig.colorbar(p3, ax=axes[2])


            plt.show()

        else:
            plt.colorbar(fd.tricontourf(self.u_sol, levels = 50))
            plt.title("Nummerical solution")
            plt.show()
        

if __name__ == "__main__":

    print("Computing h-convergence...\n")

    # Meshes resolutions
    hs = np.arange(50, 501, 50)
    error_h = []
    
    for h in hs:
        print(f"Computing for h = {h}...")

        # Mesh
        mesh = fd.UnitSquareMesh(h, h)

        # Solver
        model = PoissonSolver(mesh, P = 1)

        # Imposing true solution
        true_sol = fd.Function(model.V)
        true_sol.interpolate(fd.sin(model.x)*fd.sin(model.y))
        model.impose_true_sol(true_sol)

        # Dirichlet Boundary conditions
        DirBCs = fd.Function(model.V)
        DirBCs.interpolate(fd.sin(model.x)*fd.sin(model.y))
        model.impose_DBC(DirBCs, 1)
        model.impose_DBC(DirBCs, 2)

        # Neumann Boundary conditions
        NBC_x = fd.Function(model.V)
        NBC_y = fd.Function(model.V)

        NBC_x.interpolate(fd.cos(model.x)*fd.sin(model.y))
        NBC_y.interpolate(fd.sin(model.x)*fd.cos(model.y))

        NBCs = fd.Function(model.W)
        NBCs.vector().set_local(np.concatenate((NBC_x.vector().get_local(), NBC_y.vector().get_local())))

        model.impose_NBC(NBCs, 3)
        model.impose_NBC(NBCs, 4)

        # Solve
        model.solve()

        # Compute error
        error_h.append(fd.errornorm(true_sol, model.u_sol, norm_type="L2"))
    
    print(error_h)   

    print("Computing p-convergence...\n")

    # Setup
    Ps = np.arange(1, 7)
    error_p = []

    for P in Ps:
        print(f"Computing for P = {P}...")

        # Mesh
        mesh = fd.UnitSquareMesh(100, 100)

        # Solver
        model = PoissonSolver(mesh, P = int(P))

        # Imposing true solution
        true_sol = fd.Function(model.V)
        true_sol.interpolate(fd.sin(model.x)*fd.sin(model.y))
        model.impose_true_sol(true_sol)

        # Dirichlet Boundary conditions
        DirBCs = fd.Function(model.V)
        DirBCs.interpolate(fd.sin(model.x)*fd.sin(model.y))
        model.impose_DBC(DirBCs, 1)
        model.impose_DBC(DirBCs, 2)

        # Neumann Boundary conditions
        NBC_x = fd.Function(model.V)
        NBC_y = fd.Function(model.V)

        NBC_x.interpolate(fd.cos(model.x)*fd.sin(model.y))
        NBC_y.interpolate(fd.sin(model.x)*fd.cos(model.y))

        NBCs = fd.Function(model.W)
        NBCs.vector().set_local(np.concatenate((NBC_x.vector().get_local(), NBC_y.vector().get_local())))

        model.impose_NBC(NBCs, 3)
        model.impose_NBC(NBCs, 4)

        # Solve
        model.solve()

        # Compute error
        error_p.append(fd.errornorm(true_sol, model.u_sol, norm_type="L2"))
    print(error_p)

    error_h = np.array(error_h).reshape(-1, 1)
    error_p = np.array(error_p).reshape(-1, 1)

    #np.savetxt("./F25_Bachelor_NACA_SEM/PoissonSolver/PoissonError_h.txt", error_h)
    #np.savetxt("./F25_Bachelor_NACA_SEM/PoissonSolver/PoissonError_p.txt", error_p)
    


    # Plotting

    fig, axes = plt.subplots(1, 2, figsize = (10, 5))

    axes[0].loglog(hs, error_h, "-o")
    axes[0].set_title("h-convergence")
    axes[0].set_xlabel("h")
    axes[0].set_ylabel("Error")

    axes[1].loglog(Ps, error_p, "-o")
    axes[1].set_title("p-convergence")
    axes[1].set_xlabel("P")
    axes[1].set_ylabel("Error")

    plt.show()



