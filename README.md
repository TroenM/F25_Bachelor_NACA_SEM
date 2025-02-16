# F25_Bachelor_NACA_SEM

Last updated: 16/02

Meshing
------

mesh_library.py: Simple meshing library
    To do:
    1. Implement meshing for NACA airfoils from txt-files
    2. Consider reording algorithms for optimized node ordering.

mesh_playground.py: Testing notebook for meshing_library

Fem-solver
-------

FEM_solver.py: Contains the class defintion for an incompressible potential flow.
    To do:
        1. Implement neumann BC
        2. Implement plotting methods for non-square meshes
        3. Implement plotting methods for non-unform meshes