import os; os.system('cls||clear'); os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import torch 
import firedrake as fd

class Wave:
    def __init__(self):
        return
class Input:
    def __init__(self):
        return

def WaveProporties(Input):
    # Purpose: Compute wave proporties
    # Author: Jens Visbech, jvis@dtu.dk
    # Date: 14/02-2025 
    
    # Constants:
    Wave.g = 9.82                   # gravitational acceleration
    Wave.rho = 1024                 # density of water

    # Extract from Input class:
    Wave.kh = Input.WaveNumber                               # Non-dimensional wave number
    Wave.theta = Input.AngleOfAttack/180*np.pi               # Angle of attack

    # Wave proporties:
    Wave.L = 1                                           # length
    Wave.k = 2*np.pi/Wave.L                              # number
    Wave.h = Wave.kh/Wave.k                              # depth
    Wave.HoL = HeightOverLength(Wave.kh)                 # height over length
    Wave.HoL_max = Input.HeightOverLengthMaximum    
    Wave.H = Wave.HoL_max/100 * Wave.HoL                 # height
    Wave.A = Wave.H/2                                    # amplitude
    Wave.theta = 0                                       # angle of attack

    # Parameters for the SF routine:
    Wave.U,Wave.ubar,Wave.R,Wave.Q,Wave.cofA,Wave.cofB = StreamFunctionCoefficients(nn=25, H=Wave.H, hh=Wave.h, LL=Wave.L, gg=Wave.g, uEorSS=0, EorSS=0, nsteps=6, maxiter=2000, xtol=1e-3)

    Wave.T = Wave.L/Wave.U;                              # period
    Wave.omega = 2*np.pi/Wave.T;                         # frequency
    Wave.A = Wave.H/2                                    # amplitude

    return Wave

def WaveSolutions(Input,quantity,Wave,mesh,X,t):
    # Purpose: Compute wave solutions for linear and nonlinear waves
    # Author: Jens Visbech, jvis@dtu.dk
    # Date: 14/02-2025

    # Check if data is correct format:
    if isinstance(X, np.ndarray) == False: X = X.dat.data

    # Extract dimension and name:
    name = mesh.name
    dim = mesh._topological_dimension
    
    # Extract coordinates
    if name == "Omega":
        if dim == 2: x = X[:,0]; z = X[:,1]; y = x*0
        elif dim == 3: x = X[:,0]; y = X[:,1]; z = X[:,2]
    elif name == "FS":
        if dim == 1: x = X[:,0]; y = x*0; z = x*0
        elif dim == 2: x = X[:,0]; y = X[:,1]; z = x*0
    
    out = StreamFunctionSolution(quantity,name,Wave,x,y,z,t)
    
    return out

def HeightOverLength(kh):
    HoL = (0.141063+0.0095721*(2*np.pi/kh)+0.0077829*(2*np.pi/kh)**2)/(1+0.0788340*(2*np.pi/kh)+0.0317567*(2*np.pi/kh)**2+0.0093407*(2*np.pi/kh)**3)
    return HoL

def StreamFunctionCoefficients(nn, H, hh, LL, gg, uEorSS, EorSS, nsteps, maxiter, xtol):
    # Purpose: Compute stream function coefficients
    # Author: Jens Visbech, jvis@dtu.dk
    # Date: 14/02-2025

    UsePyTorch = True 
    if UsePyTorch: xtol = 1e-15

    global n, g, k, h, Hi, EorS, uEorS
    
    # Assign the global variables
    n = nn
    g = gg
    L = LL
    h = hh
    uEorS = uEorSS
    EorS = EorSS
    
    # Initialize some constants
    k = 2 * np.pi / L
    T0 = 2 * np.pi / np.sqrt(g * k * np.tanh(k * h))
    
    # The first step in height and the solution from linear theory
    Hi = H / nsteps
    c = L / T0
    ubar = L / T0
    R = g / (2 * k) * np.tanh(k * h)
    Q = 0
    eta = Hi / 2 * np.cos((np.arange(1, n + 2)) * L / (2 * n))
    B = np.zeros(n)
    B[0] = g * Hi / 2 * T0 / L
    
    # Initial solution
    inits = np.concatenate(([c, ubar, R, Q], eta, B))

    # Set up options for fsolve
    N = len(inits)  # Number of variables
    maxfev = maxiter * (N + 1)  # Estimate max function evaluations

    # fsolve options
    fsolve_options = {'xtol': xtol,'maxfev': maxfev}

    # Find the solution iteratively:
    if UsePyTorch:
        param0 = torch.tensor(inits, dtype=torch.float64, requires_grad=True)
        f1 = newton_solver(solver_torch, param0,xtol,maxfev); f2 = f1
        for is_ in range(2, nsteps + 1):
            Hi = is_ * H / nsteps
            inits = 2 * f2 - f1
            f = newton_solver(solver_torch, inits)
            f1 = f2
            f2 = f
        f2 = f2.detach().cpu().numpy()

    else:
        f1 = fsolve(solver, inits, **fsolve_options); f2 = np.copy(f1)
        for is_ in range(2, nsteps + 1):
            Hi = is_ * H / nsteps; 
            inits = 2 * f2 - f1
            f = fsolve(solver, inits, **fsolve_options)
            f1 = np.copy(f2)
            f2 = np.copy(f)
    
    # Extract data
    c = f2[0]
    ubar = f2[1]
    R = f2[2]
    Q = f2[3]
    eta = f2[4:4 + n + 1]
    B = f2[4 + n + 1:4 + 2 * n + 1]
    
    A = np.zeros(n)
    for i in range(1, n+1):
        A[i-1] = (2 / n) * (0.5 * (eta[0] + eta[n] * np.cos(k * i * L / 2)))
        for j in range(2, n+1):  
            A[i-1] += (2 / n) * eta[j-1] * np.cos((j - 1) * k * i * L / (2 * n))

    return c, ubar, R, Q, A, B

def solver(param):
    global n, g, k, h, Hi, EorS, uEorS

    # Unpack:
    c = param[0]  
    ubar = param[1]  
    R = param[2]  
    Q = param[3]  
    eta = param[4:4 + n + 1]  
    B = param[4 + n + 1:4 + 2 * n + 1]  

    # Compute estimates
    i = np.arange(1, n + 2); j = np.arange(1, n + 1)  

    # Element-wise operations for psia and psib
    psia = np.zeros(n + 1)
    psib = np.zeros(n + 1)
    for idx in range(n):
        psia += (B[idx] / (j[idx] * k)) * (np.sinh(j[idx] * k * eta) * np.cos(j[idx] * (i - 1) * np.pi / n))
        psib += (B[idx] / (j[idx] * k) * np.tanh(j[idx] * k * h)) * (np.cosh(j[idx] * k * eta) * np.cos(j[idx] * (i - 1) * np.pi / n))
    psi = -ubar * eta + psia + psib  

    # Element-wise operations for ua and ub
    ua = np.zeros(n + 1); ub = np.zeros(n + 1)
    for idx in range(n):
        ua += B[idx] * (np.cosh(j[idx] * k * eta) * np.cos(j[idx] * (i - 1) * np.pi / n))
        ub += B[idx] * np.tanh(j[idx] * k * h) * (np.sinh(j[idx] * k * eta) * np.cos(j[idx] * (i - 1) * np.pi / n))
    u = -ubar + ua + ub  

    # Element-wise operations for wa and wb
    wa = np.zeros(n + 1); wb = np.zeros(n + 1)
    for idx in range(n):
        wa += B[idx] * (np.sinh(j[idx] * k * eta) * np.sin(j[idx] * (i - 1) * np.pi / n))
        wb += B[idx] * np.tanh(j[idx] * k * h) * (np.cosh(j[idx] * k * eta) * np.sin(j[idx] * (i - 1) * np.pi / n))
    w = wa + wb  

    # Setup equations
    eq1 = eta[0] - eta[n] - Hi  
    eq2 = eta[0] + eta[n] + 2 * np.sum(eta[1:n])  
    if EorS == 0:
        eq3 = uEorS + ubar - c  
    else:
        if k * h < 24:
            eq3 = (uEorS + ubar - c) * h - Q 
        else:
            eq3 = (uEorS + ubar - c) * h 

    bc1 = psi - Q 
    bc2 = g * eta + 0.5 * (u**2 + w**2) - R  

    # Return values
    return np.concatenate(([eq1], [eq2], [eq3], bc1, bc2))

def solver_torch(param):
    c     = param[0]
    ubar  = param[1]
    R     = param[2]
    Q     = param[3]
    eta   = param[4:4 + n + 1]
    B     = param[4 + n + 1:4 + 2 * n + 1]

    i = torch.arange(1, n + 2, dtype=param.dtype, device=param.device).unsqueeze(0)  # shape (1, n+1)
    j = torch.arange(1, n + 1, dtype=param.dtype, device=param.device).unsqueeze(1)  # shape (n, 1)

    eta_b = eta.unsqueeze(0)  # shape (1, n+1)

    psia = (B / (j.squeeze() * k)).unsqueeze(1) * torch.sinh(j * k * eta_b) * torch.cos(j * (i - 1) * torch.pi / n)
    psib = (B / (j.squeeze() * k) * torch.tanh(j.squeeze() * k * h)).unsqueeze(1) * \
           torch.cosh(j * k * eta_b) * torch.cos(j * (i - 1) * torch.pi / n)
    psi = -ubar * eta + psia.sum(dim=0) + psib.sum(dim=0)

    ua = (B.unsqueeze(1) * torch.cosh(j * k * eta_b) * torch.cos(j * (i - 1) * torch.pi / n)).sum(dim=0)
    ub = (B.unsqueeze(1) * torch.tanh(j.squeeze() * k * h).unsqueeze(1) *
          torch.sinh(j * k * eta_b) * torch.cos(j * (i - 1) * torch.pi / n)).sum(dim=0)
    u = -ubar + ua + ub

    wa = (B.unsqueeze(1) * torch.sinh(j * k * eta_b) * torch.sin(j * (i - 1) * torch.pi / n)).sum(dim=0)
    wb = (B.unsqueeze(1) * torch.tanh(j.squeeze() * k * h).unsqueeze(1) *
          torch.cosh(j * k * eta_b) * torch.sin(j * (i - 1) * torch.pi / n)).sum(dim=0)
    w = wa + wb

    eq1 = eta[0] - eta[n] - Hi
    eq2 = eta[0] + eta[n] + 2 * eta[1:n].sum()
    eq3 = uEorS + ubar - c if EorS == 0 else (uEorS + ubar - c) * h - Q if k * h < 24 else (uEorS + ubar - c) * h

    bc1 = psi - Q
    bc2 = g * eta + 0.5 * (u**2 + w**2) - R

    return torch.cat([eq1.view(1), eq2.view(1), eq3.view(1), bc1, bc2])

def newton_solver(func, x0, tol=1e-15, max_iter=10000):
    # Newton solver using PyTorch autograd:

    x = x0.clone().detach().requires_grad_(True)

    for i in range(max_iter):
        f = func(x)
        norm = torch.norm(f, p=float('inf')).item()
        # print(f"[{i:02d}] ||F(x)|| = {norm:.3e}")
        if norm < tol:
            return x.detach()

        J = []
        for f_i in f:
            grad = torch.autograd.grad(f_i, x, retain_graph=True)[0]
            J.append(grad)
        J = torch.stack(J)  # shape: (len(f), len(x))

        delta = torch.linalg.solve(J, f)
        x = (x - delta).detach().requires_grad_(True)

    raise RuntimeError("Newton method did not converge.")

def StreamFunctionSolution(quantity,name,Wave,x,y,z,t):
    # Purpose: Compute stream function solution
    # Author: Jens Visbech, jvis@dtu.dk
    # Date: 14/02-2025

    # Assumptions:
    # - Constant depth, h
    # - Waves only travel in the x-direction, theta = 0.

    # Unpack parameters:
    cofA, cofB, U, k, U, ubar, h = Wave.cofA, Wave.cofB, Wave.U, Wave.k, Wave.U, Wave.ubar, Wave.h

    # Change frame of reference:
    x = x - U*t

     # Precompute index array and allocate output
    j = np.arange(1, len(cofA) + 1)
    out = np.zeros(len(x)) 
    eta = np.zeros(len(x)) 
    eta_t = np.zeros(len(x)) 

    for i in range(len(x)):
        if name == "FS":

            # Compute free surface
            eta[i] = np.cos(k * x[i] * j) @ cofA

            if   quantity == "eta":      out[i] = eta[i]
            elif quantity == "eta_x":    out[i] = -k * np.sin(k * x[i] * j) @ (cofA*j)
            elif quantity == "eta_t":    out[i] = k * U * np.sin(k * x[i] * j) @ (cofA*j)
            elif quantity == "u_FS":     out[i] = (U - ubar) + ((np.cosh(k * eta[i] * j) + np.sinh(k * eta[i] * j) * np.tanh(k * h * j) ) * np.cos(k * x[i] * j)) @ cofB
            elif quantity == "v_FS":     exit('v_FS not implemented jet!')
            elif quantity == "w_FS":     out[i] = ((np.sinh(k * eta[i] * j) + np.cosh(k * eta[i] * j) * np.tanh(k * h * j)) * np.sin(k * x[i] * j)) @ cofB
            elif quantity == "phi_FS":   out[i] = (U - ubar) * x[i] + ((np.cosh(k * eta[i] * j) + np.sinh(k * eta[i] * j) * np.tanh(k * h * j)) * np.sin(k * x[i] * j)) @ (cofB / (j * k))
            elif quantity == "phi_t_FS": eta_t[i] = k * U * np.sin(k * x[i] * j) @ (cofA*j); out[i] = -(U-ubar)*U + (k * eta_t[i] * j * np.sinh(k*eta[i]*j) + k *eta_t[i] * j *np.cosh(k*eta[i]*j) * np.tanh(k*h*j)) * np.sin(k*x[i] *j) @ (cofB / (j * k)) - (np.cosh(k*eta[i]*j) + np.sinh(k*eta[i]*j) * np.tanh(k*h*j)) * k*U*j * np.cos(k*x[i]*j) @ (cofB / (j * k))
            else: exit('Error: not correct input in StreamFunctionSolution()')

        if name == "Omega":

            if quantity == "u": out[i] = (U - ubar) + ((np.cosh(k * z[i] * j) + np.sinh(k * z[i] * j) * np.tanh(k * h * j)) * np.cos(k * x[i] * j)) @ cofB
            elif quantity == "v": exit('v not implemented jet!')
            elif quantity == "w": out[i] = ((np.sinh(k * z[i] * j) + np.cosh(k * z[i] * j) * np.tanh(k * h * j)) * np.sin(k * x[i] * j)) @ cofB
            elif quantity == "phi": out[i] = (U - ubar) * x[i] + ((np.cosh( k*(z[i]+h)*j )/np.cosh(k*h*j ))* np.sin(k*x[i]*j) ) @(cofB/(j*k));#out[i] = (U - ubar) * x[i] + ((np.cosh(k * z[i] * j) + np.sinh(k * z[i] * j) * np.tanh(k * h * j)) * np.sin(k * x[i] * j)) @ (cofB / (j * k))
            elif quantity == "phi_t": out[i] = -U * ((U - ubar) + ((np.cosh(k * z[i] * j) + np.sinh(k * z[i] * j) * np.tanh(k * h * j)) * np.cos(k * x[i] * j)) @ cofB) 
            else: exit('Error: not correct input in StreamFunctionSolution()')

    return out

def BuildFunctionSpacesAndMore(mesh,basis,P):
    # Purpose: Build function space, vector function space, and coordinates of order P.
    # Author: Jens Visbech, jvis@dtu.dk
    # Date: 14/02-2025

    # Function space:
    V = fd.FunctionSpace(mesh,basis,P)

    # Vector functions space:
    W = fd.VectorFunctionSpace(mesh,basis,P)

    # Coordinates of order P
    X = fd.Function(W).interpolate(mesh.coordinates)

    return V,W,X

# Set inputs:
Input.WaveNumber = np.pi # [1; 2*np.pi] non-dimentional.
Input.HeightOverLengthMaximum = 50 # [1; 90] percent.
Input.AngleOfAttack = 0 # Always 0 degrees.

Nx = 10 # Number of elements in x-direction
Nz = 10 # Number of elements in z-direction
Input.Order = 4 # Order of the polynomial

# Compute wave properties:
Wave = WaveProporties(Input) # Class for wave proporties
# Build meshes:
mesh_FS = fd.PeriodicIntervalMesh(ncells=Nx,length=Wave.L, name='FS')
mesh_Omega = fd.ExtrudedMesh(mesh=mesh_FS,layers=Nz,layer_height=Wave.h/Nz, extrusion_type='uniform', name='Omega')
mesh_Omega.coordinates.dat.data[:, 1] -= Wave.h # Shift mesh down 
W_FS_DG_P1 = fd.VectorFunctionSpace(mesh_FS, "DG", 1, dim=2); x, = fd.SpatialCoordinate(mesh_FS); mesh_FS = fd.Mesh(fd.Function(W_FS_DG_P1).interpolate(fd.as_vector([x, x*0])), name='FS') # Immerse Gamma_FS in Omega dimension:

# Build function spaces:
V_Omega, W_Omega, X_Omega = BuildFunctionSpacesAndMore(mesh_Omega,"CG",Input.Order)
V_FS,    W_FS,    X_FS    = BuildFunctionSpacesAndMore(mesh_FS,"CG",Input.Order)

# Plotting of some free surface quantity at time, t:
t = 0 # [s]. Could also be t = Wave.T/2 (half wave period)
quantity = "eta" # ["eta", "phi_FS", "u_FS", etc... (see StreamFunctionSolution())]
eta = fd.Function(V_FS); eta.dat.data[:] = WaveSolutions(Input,quantity,Wave,mesh_FS,X_FS,t)

plt.figure(1)
plt.plot(X_FS.dat.data[:,0],eta.dat.data[:],'or')
plt.xlabel('x [m]'); plt.ylabel('eta [m]'); plt.grid()
plt.title('Free surface elevation at t = %s' % t); plt.show()

# Plotting of some fluid domain quantity at time, t:
t = Wave.T/2
quantity = "phi" # ["u", "v", "w", "phi", etc... (see StreamFunctionSolution())]
phi = fd.Function(V_Omega); phi.dat.data[:] = WaveSolutions(Input,quantity,Wave,mesh_Omega,X_Omega,t)

plt.figure(2)
ax = plt.subplot(projection='3d')
ax.scatter(X_Omega.dat.data[:,0], X_Omega.dat.data[:,1], phi.dat.data[:], 'or')
plt.xlabel('x [m]'); plt.ylabel('y [m]'); plt.title('Fluid domain quantity at t = %s' % t)
plt.grid()
plt.show()

