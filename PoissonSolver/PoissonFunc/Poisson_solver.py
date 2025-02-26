import firedrake as fd
from firedrake.__future__ import interpolate  # Import updated interpolate behavior

def solve_poisson_2d(mesh : fd.Mesh, p : int, rhs: callable, dirichlet: list, neumann: list, BC: dict[callable]):
    if len(set(dirichlet) | set(neumann)) != len(dirichlet) + len(neumann) or len(set(dirichlet) | set(neumann)) != 4:
        raise ValueError(f"Dirichlet and Neumann boundary conditions must cover all boundaries only once")

    V = fd.FunctionSpace(mesh, "CG", p)
    u = fd.TrialFunction(V)
    v = fd.TestFunction(V)
    x, y = fd.SpatialCoordinate(V)

    # Dirichlet Boundary Conditions
    bcs = []
    for i in dirichlet:
        boundary_function = fd.Function(V)
        boundary_function.interpolate(BC[i](x, y))
        bcs.append(fd.DirichletBC(V, boundary_function, i))

    # Define rhs
    f = fd.Function(V)
    f.interpolate(rhs(x, y))

    # Define Weak Form
    a = fd.inner(fd.grad(u), fd.grad(v)) * fd.dx

    # Neumann Boundary Conditions
    neumann_sum = 0
    for i in neumann:
        neumann_function = fd.Function(V)
        neumann_function.interpolate(BC[i](x, y))
        neumann_sum += (neumann_function * v) * fd.ds(i)
    
    L = (-f * v) * fd.dx + neumann_sum
    u_sol = fd.Function(V)

    # Solve
    fd.solve(a == L, u_sol, bcs=bcs)

    return u_sol