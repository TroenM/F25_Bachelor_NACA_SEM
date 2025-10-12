import os;
os.environ["OMP_NUM_THREADS"] = "1"
os.system('cls||clear')

from MeshEssentials import *

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
rho = 1.225

# Mesh settings
airfoilNumber = "0012"
alpha_deg = 45 # Angle of attack in degrees
centerOfAirfoil = (0.0,0)
circle = True
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

def __initialise_relevant_mesh_data__():
    alpha = np.deg2rad(alpha_deg)
    a = 1
    b = 1 if circle else int(airfoilNumber[2:])/100
    mesh = naca_mesh(airfoil=airfoilNumber, n_airfoil = nAirfoil,alpha = alpha, xlim = xlim, ylim=ylim, center_of_airfoil = centerOfAirfoil)
    V = fd.FunctionSpace(mesh, "CG", P)

    fs_indecies = V.boundary_nodes(4)
    W = fd.VectorFunctionSpace(mesh, "CG", P)
    coordsFS = (fd.Function(W).interpolate(mesh.coordinates).dat.data)[fs_indecies,:]
    return alpha, a, b, mesh, V, fs_indecies, W, coordsFS

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

def __normalize_vector__(vector : np.ndarray) -> np.ndarray:
    if type(vector) != np.ndarray or np.linalg.norm(vector) == 0:
        raise TypeError("The vector has to be a numpy array of length more than 0")
    return vector/np.linalg.norm(vector)

def __findAirfoilDetails__(airfoilNumber : str, nAirfoil : int, alpha : float, centerOfAirfoil : tuple) -> tuple: #Find the leading and trailing edge of the airfoil
    '''
    Find the leading and trailing edge of the airfoil by centering and rotating the airfoil to alpha = 0 
    and finding the min and max x-coordinates, then rotating back to the original angle of attack and shifting back to the original center.

    THIS IS ONLY NECESSARY ONCE, AT THE START OF THE SIMULATION.
    '''
    # Calculate airfoil coordinates
    naca_coords = naca_4digit(airfoilNumber,nAirfoil, alpha, centerOfAirfoil)
    # Gathering position of Leading edge, Trailing edge, 
    # the first point on the bottom surface from the trailing edge (p1) and the first on the top surface (pn)
    TE = naca_coords[0]
    LE = naca_coords[nAirfoil//2]
    p1 = naca_coords[1]
    pn = naca_coords[-1]

    # Calculate a normalised vector going from p1 to TE and a normalizes vector going from pn to TE
    v1 = __normalize_vector__(TE - p1)
    vn = __normalize_vector__(TE - pn)

    # Using these vectors to calculate the normalized vector that is orthorgonal 
    # to the direction of the trailing edge (vPerp)
    vPerp = __normalize_vector__(v1 - vn)

    # Using vPerp to find a point that is just outside the trailing edge in the direction of the trailing edge
    pointAtTE = TE + np.array([vPerp[1], -vPerp[0]])/70

    return LE, TE, vPerp, pointAtTE

def __FBCS__(Gammas : list) -> float: # Applies the FBCS-scheme discussed in the report
    # Adaptive stepsize controller for Gamma described in the report
    if len(Gammas) == 1:
        return Gammas[-1]/c0
    else:
        a = (Gammas[-1]/c0) / Gammas[-2]
        c0 *= (1-a)
        return Gammas[-1]/c0

def __compute_vortex_strength__(vPerp, VelocityAtTE, PointAtTE) -> float:
    """
    Computes the vortex strength for the given iteration
    """
    # Get the coordinates of the point just outside of the trailing edge
    p_x = PointAtTE[0]
    p_y = PointAtTE[1]

    # Translating the trailing edge coordinates to have the center at the origin
    x_t = p_x - centerOfAirfoil[0]
    y_t = p_y - centerOfAirfoil[1]

    # Rotating the trailing edge coordinates to align with "not-rotated" coordinates
    x_bar = x_t * np.cos(alpha) - y_t * np.sin(alpha)
    y_bar = x_t * np.sin(alpha) + y_t * np.cos(alpha)

    # Computing vortex at trailing edge without  the scaling factor Gamma/ellipse_circumference
    Wx = -(y_bar/b) / (x_bar**2/a + y_bar**2/b)
    Wy = (x_bar/a) / (x_bar**2/a + y_bar**2/b)

    # Rotating the vortex vector at the trailing edge clockwise by alpha,
    # in order to mimic that vectors orientation in the rotated vortex field
    Wx_rot = Wx * np.cos(-alpha) - Wy * np.sin(-alpha)
    Wy_rot = Wx * np.sin(-alpha) + Wy * np.cos(-alpha)

    # Calculating the circumference of the ellipse (scaling factor)
    ellipse_circumference = np.pi*(3*(a+b) - np.sqrt(3*(a+b)**2+4*a*b))

    # Computing the vortex strength Gamma
    Gamma = -ellipse_circumference*(vPerp[0]*VelocityAtTE[0] + vPerp[1]*VelocityAtTE[1])/(Wx_rot*vPerp[0] + Wy_rot*vPerp[1])

    return Gamma

def __compute_vortex__(Gamma, center_of_vortex, vortex) -> fd.Function:
    """
    Computes the vortex field for the given vortex strength
    """
    # Define alpha, a and b as firedrake coordinates
    alpha = fd.Constant(alpha)
    a = fd.Constant(a)
    b = fd.Constant(b)
    
    # Gather coordinates from fd mesh
    fd_x, fd_y = fd.SpatialCoordinate(mesh)

    # Translate coordinates such that they have their center in origo
    x_translated = fd_x - center_of_vortex[0]
    y_translated = fd_y - center_of_vortex[1]
    
    # rotate the coordinates such that they are aranged as "unrotated coordinates"
    x_bar = (x_translated) * fd.cos(alpha) - (y_translated) * fd.sin(alpha)
    y_bar = (x_translated) * fd.sin(alpha) + (y_translated) * fd.cos(alpha)
    
    # Calculate the approximated circumference of the ellipse (scaling factor)
    ellipse_circumference = fd.pi*(3*(a+b) - fd.sqrt(3*(a+b)**2+4*a*b))

    # Compute the unrotated elliptical vortex field onto the "unrotated" coordinates
    u_x = -Gamma / ellipse_circumference * y_bar/b / ((x_bar/a)**2 + (y_bar/b)**2)
    u_y = Gamma / ellipse_circumference * x_bar/a / ((x_bar/a)**2 + (y_bar/b)**2)

    # Rotate the final vectors in the vortex field
    u_x_final = u_x * fd.cos(-alpha) - u_y * fd.sin(-alpha)
    u_y_final = u_x * fd.sin(-alpha) + u_y * fd.cos(-alpha)

    # project the final vectorfunction onto the original coordinates of the mesh
    vortex.project(fd.as_vector([u_x_final, u_y_final]))

    return vortex

def applyKuttaCondition():
    """
    Applies one iteration of the Kutta condition by adding and adjusting for a vortex at centerOfAirfoil.
    """
    # Find the circulation strength Gamma that makes the velocity at the trailing edge zero
    # This is done using a simple bisection method
    LE, TE, vPerp, PointAtTE = __findAirfoilDetails__(mesh, alpha)
    vortex = fd.Function(W, name="vortex")
    # Calculate initial velocity
    phi = PoissonSolver(mesh, V)
    velocity = fd.Function(W).interpolate(fd.grad(phi))
    Gammas = []

    for it, _ in enumerate(range(maxItKutta + 1)):
        # Gathering the velocity at the point a little out from the trailing edge
        velocityAtTE = velocity.at(PointAtTE)

        # Using the vortex strength a little out from the trailing estimate a new vortex strength
        Gamma = __compute_vortex_strength__(vPerp, velocityAtTE, PointAtTE)
        Gammas.append(Gamma)

        # Use an adaptive method to do feedback controlled scaling of the strength of the vortex
        Gamma = __FBCS__(Gammas)
        
        # Redifine Gamma as the scaled version
        Gammas[-1] = Gamma

        # Compute the vortex
        vortex = __compute_vortex__(Gamma, centerOfAirfoil, vortex)

        # Sum velocity and the vortex to add circulation to the flow
        velocity += vortex

        # Computing the boundary correction
        BoundaryCorrection = __compute_boundary_correction__(vortex)
        velocity += BoundaryCorrection

    # Compute lift based on circulation given the formula in the Kutta Jacowski theorem
    lift = -Gamma * V_inf * rho
    lift_coeff = lift / (1/2 * rho * V_inf**2)

    # Compute pressure coefficients
    C_p = __compute_pressure_coefficients(velocity)
    return velocity, C_p, Gammas, lift_coeff

#=================================================================#
#======================= Free Surface ============================#
#=================================================================#

def weak1DWaveEquations():
    return



#=================================================================#
#=========================== Main Loop ===========================#
#=================================================================#
if __name__ == "__main__":
    LE, TE = findAirfoilDetails(mesh, alpha)



    naca_coords = naca_4digit(airfoilNumber,nAirfoil, alpha, centerOfAirfoil)
    print(naca_coords[0])
    print(naca_coords[1])
    print(naca_coords[-1])
    print(LE)
    print(TE)
    # plt.scatter(naca_coords[:, 0], naca_coords[:, 1])
    # plt.scatter(LE[0], LE[1], color='red') # Leading edge
    # plt.scatter(TE[0], TE[1], color='yellow') # Trailing edge
    # plt.axis('equal')
    # plt.show()

