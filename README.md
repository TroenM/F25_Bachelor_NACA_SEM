# F25_Bachelor_NACA_SEM
Last updated: 12. Aug 2025
The project is no longer maintained.

This is the code for the bachelor thesis: Spectral element methods for potential flow around bodies.
By Magnus Troen and Morten Rønn Østergaard
Supervised by Allan P. Ensig-Karap
Co-supervised by Jans Visbech
DTU compute

The project builds a free surface solver for a 2d naca-airfoil submerged just under the surface of steady water.

Note that both `PoissonSolver` and `PotentialFlowSolver` are stable and can be proven to converge. `FsSolver` is unstable and has not been proven to converge.

Requirements
---
Firedrake 0.14.dev0
gmsh 4.13.1
meshio 5.3.5
vtk 9.4.1
matplotlib 3.10.0


Solver Eco system
---
The complete free surface solver consists of 4 layers. Each outer layer can call upon an inner layer. The 
1. Meshing
   - Constructs a rectangular unstructured triangulated 2d mesh with a naca-airfoil in the middle.
   - Updates the surface for free surface modelling
   - Code found in Meshing/mesh_library.py
2. Potential solver
    - Solves the governing equations for potential flow.
    - Code found in PoissonSolver/PoissonCLS/poisson_solver.py
3. Channel Flow Solver
    - Imposes the Kutta condition through iterative calls to the Poisson solver.
    - Code found in Potential_flow_solver/PotentialFlowSolverCLS/PotentialFlowSolver.py
4. Free surface solver
    - Impose governing equations for free surface behavior through iterative calls to the channel flow solver
    - Code found in Fs_solver/fs_solver.py

`Miscellaneous Folders`
---
1. Visualisation
    - Collection of .pvd-files and other image formats used for the report.

2. FEM-solver
    Contains:
    - FEM_solver.py
        - Our own implementation of a FEM solver for poisson problems
    - FEM_playground.ipynb
        - A jupyter-notebook for quick testing and visualizations.

3. HPC_Setup
    Contains:
    - HPC.py
        - The entire free surface solver for easy job submission on a cluster
    - FS_submit.sh
        - A submish script for the free surface solver

