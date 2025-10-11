import os;
os.environ["OMP_NUM_THREADS"] = "1"
os.system('cls||clear')

from MeshEssentials import naca_mesh, shift_surface

import firedrake as fd
from firedrake.pyplot import tripcolor
import numpy as np
import matplotlib.pyplot as plt
from time import time

"""
IMPORTANT: The boundaries should be indexed as follows:
1: Inflow
2: Outflow
3: Bed
4: Free Surface
5: Airfoil
"""


#================================================================#
#======================= Hyper parameters =======================#
#================================================================#

P = 1 # Polynomial_Order
V_inf = fd.as_vector((1.0, 0.0)) # Free-stream velocity

# Mesh settings
airfoilNumber = "0012"
alpha = 45 # Angle of attack in degrees
centerOfAirfoil = (0.0,0)
xlim = (-7, 13)
ylim = (-4, 2)
nIn = 20
nOut = 20
nBed = 50
nFS = 50 
nAirfoil = 100

# Solver settings
maxItKutta = 50
maxItFreeSurface = 50
c0 = 7 # Initial Damping of Vortex Strength


#=================================================================#
#================== Function Spaces and Meshes ===================#
#=================================================================#
mesh = naca_mesh(airfoil=airfoilNumber, n_airfoil = nAirfoil,alpha = alpha, xlim = xlim, ylim=ylim, center_of_airfoil = centerOfAirfoil)
V = fd.FunctionSpace(mesh, "CG", P)

fs_indecies = V.boundary_nodes(4)
W = fd.VectorFunctionSpace(mesh, "CG", P)
coordsFS = (fd.Function(W).interpolate(mesh.coordinates).dat.data)[fs_indecies,:]

#=================================================================#
#======================== Poisson Solver =========================#
#=================================================================#
def PoissonSolver(mesh, V, rhs = fd.Constant(0.0), DBC=[], NBC=[(1, V_inf), (2, V_inf)]):
    """ Solves the Poisson equation:
        -Δφ = rhs in Ω
        φ = DBCfunc on DBCidx
        ∂φ/∂n = NBCfunc on NBCidx

        Inputs:
        mesh: Firedrake mesh
        V: Function space for potential φ
        rhs: Right-hand side function (fd.Function or fd.Constant)
        DBC: List of tuples [(bcidx, DBCfunc), ...] for Dirichlet BCs
        NBC: List of tuples [(bcidx, NBCfunc), ...] for Neumann BCs
            - NBCfunc should be a vector, eg. (fd.as_vector([V_inf, 1]), fd.as_vector([V_inf, 2])) for far-field conditions
        Outputs:
        phi: Solution potential (fd.Function in V)
        """
    v = fd.TestFunction(V)
    phi = fd.TrialFunction(V) 
    a = fd.inner(fd.grad(phi), fd.grad(v)) * fd.dx
    L = rhs*v*fd.dx

    DBCs = []
    for _ in DBC:
        bcidx, DBCfunc = _
        DBCs.append(fd.DirichletBC(V, DBCfunc, bcidx)) 
    
    for _ in NBC:
        bcidx, NBCfunc = _
        L += fd.dot(NBCfunc, fd.FacetNormal(mesh)) * v * fd.ds(bcidx) # Set NBC = [fd.as_vector(V_inf, 0)]*2 for far-field

    phi = fd.Function(V) # Consider whether phi should be an input instead.
    fd.solve(a == L, phi, bcs=DBCs)

    if len(DBCs) == 0: # Normalize phi if there are no Dirichlet BCs
        phi -= np.min(phi.dat.data)
    
    # Do not define u here, it will not be accessible for VTK output
    #u = fd.Function(W).interpolate(fd.grad(phi))

    return phi

#=================================================================#
#======================= Kutta Condition =========================#
#=================================================================#

def findLEandTE(alpha): #Find the leading and trailing edge of the airfoil
    '''
    Find the leading and trailing edge of the airfoil by centering and rotating the airfoil to alpha = 0 
    and finding the min and max x-coordinates, then rotating back to the original angle of attack and shifting back to the original center.

    THIS IS ONLY NECESSARY ONCE, AT THE START OF THE SIMULATION.
    '''
    
    # Extract airfoil coordinates from mesh
    naca_coords = mesh.coordinates.dat.data[V.boundary_nodes(5)]
    
    alphaRad = np.deg2rad(alpha)
    R = np.array([[np.cos(alphaRad), -np.sin(alphaRad)],
                  [np.sin(alphaRad),  np.cos(alphaRad)]])
    
    # Center and rotate airfoil to alpha = 0
    naca_coords_centered = naca_coords - centerOfAirfoil
    naca_coords_alpha0 = naca_coords_centered @ R.T
    
    # Find Leading and Trailing edge at alpha = 0
    LE0 = naca_coords_alpha0[np.argmin(naca_coords_alpha0[:,0]), :] # Leading edge at alpha = 0
    TE0 = naca_coords_alpha0[np.argmax(naca_coords_alpha0[:,0]), :] # Trailing edge at alpha = 0
    
    # Rotate back to original angle of attack and shift back to original center
    LE = (LE0 @ R) + centerOfAirfoil
    TE = (TE0 @ R) + centerOfAirfoil

    return LE, TE

def __compute_updated_Gamma__(Gammas : list) -> float:
    # Adaptive stepsize controller for Gamma described in the report
    if len(Gammas) == 1:
        return Gammas[-1]/c0
    else:
        a = (Gammas[-1]/c0) / Gammas[-2]
        c1 = c0 * (1-a)
        return Gammas[-1]/c1

def applyKuttaCondition():
    """
    Applies one iteration of the Kutta condition by adding and adjusting for a vortex at centerOfAirfoil.
    """
    LE, TE = findLEandTE(alpha)
    # Find the circulation strength Gamma that makes the velocity at the trailing edge zero
    # This is done using a simple bisection method 
    Gammas = []
    for it, _ in enumerate(range(maxItKutta + 1)):
        # Start time
        time_it = time()

        # Computing the vortex strength at the point a little out from the trailing edge
        vte = velocity.at(p_te_new)

        # Using the vortex strength a little out from the trailing estimate a new vortex strength
        Gamma = compute_vortex_strength(v12, vte, p_te_new)
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
    if it == maxItKutta:
        print(f"PoissonSolver did not converge in {it} iterations")
        print(f"\t dGamma: {dgamma}")
        print(f"\t dot product: {np.dot(velocity.at(p_te_new), v12)}")
        print(f"\t Total time: {time() - time_total}")
        print(f"\n")

    Gammas.append(Gamma)

    # Use an adaptive method to do feedback controlled scaling of the strength of the vortex
    Gamma = __compute_updated_Gamma__(Gammas)
    
    # Redifine Gamma as the scaled version
    Gammas[-1] = Gamma

    return

#=================================================================#
#======================= Free Surface ============================#
#=================================================================#

def weak1DWaveEquations():
    return



#=================================================================#
#=========================== Main Loop ===========================#
#=================================================================#
if __name__ == "__main__":
    LE, TE = findLEandTE(alpha)



    naca_coords = mesh.coordinates.dat.data[V.boundary_nodes(5)]
    plt.scatter(naca_coords[:, 0], naca_coords[:, 1])
    plt.scatter(LE[0], LE[1], color='red') # Leading edge
    plt.scatter(TE[0], TE[1], color='yellow') # Trailing edge
    plt.axis('equal')
    plt.show()

