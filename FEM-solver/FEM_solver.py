import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import meshio

class PotentialFlowSolver_FEM():
    """
    Finite element solver for incompressible potential flow 

    Pipline:
    --------
    1. Initialize the FEM_solver object with a mesh object.
    2. Impose boundary conditions on the system.
    3. Solve the system.
    
    Attributes
    ----------
    mesh : meshio.Mesh
        A mesh object containing the mesh information.
    EtoV : np.ndarray
        2D array containing the element to vertex matrix.
    points : np.ndarray
        2D array containing the point coordinates.
    boundary_indices : np.ndarray
        An array containing the boundary indices.

    f1 : callable
        Boundary condition function for the first boundary.
    f2 : callable
        Boundary condition function for the second boundary.
    f3 : callable
        Boundary condition function for the third boundary.
    f4 : callable
        Bondary condition function for the fourth boundary.
    
    N : int
        Number of elements in the mesh.
    M : int
        Number of vertices in the mesh.
    A : sps.csr_matrix
        The coefficient matrix.
    b : np.ndarray
        The right-hand side vector.
    u : np.ndarray
        The solution vector.
    

    Initialization Methods
    -------
    __init__(mesh : meshio.Mesh)
        Initializes the FEM_solver object.
            - Extracts the element to vertex matrix (EtoV), point coordinates and boundary indices from a mesh object.
            - Initializes the coefficient matrix A, the right-hand side vector b and the solution vector u.
            - Constructs the initial system of equations.
    
    get_mesh_info(mesh : meshio.Mesh) -> np.ndarray:
        Extracs the Element to vertex matrix (EtoV), point coordinates and boundary indices from a mesh object.
    
    Construction Methods
    -------
    construct_initial_system()
        Constructs the initial system of equations, saves the coefficient matrix in A
             and the right-hand side vector in b.
    
    compute_k(element: int) -> np.ndarray
        Computes the element in element matrix for a given element.
        (Only supports triangular elements currently)
             
    compute_triangular_k(element: int) -> np.ndarray:
        Computes the element in element matrix for a triangular element. 
    
    
    Boundary Condition Methods
    -------
    impose_BC(BC : int, BC_type : str, BC_func : callable)
        Impose boundary conditions on the system. And saves BC_func in the corresponding boundary condition attribute.

    impose_Dirichlet_BC(BC : int, BC_func : callable)
        Impose Dirichlet boundary conditions on the system. And saves BC_func in the corresponding boundary condition attribute.
    
    impose_Neumann_BC(BC : int, BC_func : callable)
        Impose Neumann boundary conditions on the system. And saves BC_func in the corresponding boundary condition attribute.
        
    Solution Methods
    -------
    solve()
        Solves the finite element problem.
    

    

    """


    ##################### ATTRIBUTES #####################
    mesh: meshio.Mesh
    EtoV: np.ndarray
    points: np.ndarray
    boundary_indices: np.ndarray

    f0: callable
    f1: callable
    f2: callable
    f3: callable

    N: int
    M: int
    A: sps.csr_matrix
    b: np.ndarray
    u: np.ndarray


    ##################### INITIALIZATION #####################
    def __init__(self, mesh: meshio.Mesh):
        """
        Initializes the FEM_solver object.

        Parameters
        ----------
        mesh : meshio.Mesh
            A mesh object containing the mesh information.
        """
        self.mesh = mesh
        self.EtoV, self.points, self.boundary_indices = self.get_mesh_info(mesh)
        self.points = self.points[:, :2]
        self.N = self.EtoV.shape[0]
        self.M = self.points.shape[0]
        self.A = np.zeros((self.M, self.M))
        self.b = np.zeros(self.M)
        self.u = np.zeros(self.M)

        self.construct_initial_system()


    def get_mesh_info(self, mesh: meshio.Mesh) -> np.ndarray:
        """
        Extracs the Element to vertex matrix (EtoV), point coordinates and boundary indices from a mesh object.

        Parameters
        ----------
        mesh : meshio.Mesh
            A mesh object containing the mesh information.
        
        Returns
        -------
        EtoV : np.ndarray 
            2D array containing the element to vertex matrix.

            Format:
            [node 0, node 1, ..., node n-1] for element 0
            [node 0, node 1, ..., node n-1] for element 1
            ...
        points : np.ndarray
            3D array containing the point coordinates.

            Format:
            [x, y, z] for point 0
            [x, y, z] for point 1
            ...
        boundary_indices : np.ndarray
            An array containing the boundary indices.

            Format:
            [node 0, index 1, ..., index n-1] for boundary 0
            [node 0, index 1, ..., index n-1] for boundary 1
            ... 
        """

        # Extract the element to vertex matrix (Update to support mixed meshes)
        EtoV = mesh.cells_dict["triangle"]

        # Extract the point coordinates
        points = mesh.points

        # Extract the boundary indices
        boundary_indices = mesh.cell_sets["boundary"]

        return EtoV, points, boundary_indices
    

    ##################### Construction Methods #####################

    def compute_triangular_k(self, EtoV: np.ndarray, r, s) -> float:
        """
        Computes the element in element matrix for a triangular element. 
        (Remember to multiply with the area of the element when constructing the global matrix)

        Parameters
        ----------
        element : np.ndarray
            EtoV matrix for element.
        
        Returns
        -------
        k : float
            Element of the element matrix.
        """

       # Get the coordinates of the vertices
        coords = self.points[EtoV]

        Delta = 1/2 * (coords[1, 0]*coords[2, 1] - coords[1,1]*coords[2, 0]
                           -(coords[0, 0]*coords[2, 1] - coords[0,1]*coords[2, 0])
                           + coords[0, 0]*coords[1, 1] - coords[0,1]*coords[1, 0])

        # (i,j,k) = (0,1,2), (1,2,0), (2,0,1)
        br = coords[(r+1)%3,1] - coords[(r+2)%3,1]
        bs = coords[(s+1)%3,1] - coords[(s+2)%3,1]

        cr = coords[(r+2)%3,0] - coords[(r+1)%3,0]
        cs = coords[(s+2)%3,0] - coords[(s+1)%3,0]

        return 1/(4*np.abs(Delta))* (br*bs + cr*cs)
    
    def compute_k(self, EtoV: np.ndarray, r, s) -> float:
        """
        Computes the element in element matrix for a given element

        Parameters
        ----------
        element : np.ndarray
            EtoV matrix for the element.
        
        Returns
        -------
        k : float
            Element of the element matrix.
        """
        # Get the vertices of the element

        if len(EtoV) == 3:
            return self.compute_triangular_k(EtoV, r, s)
        else:
            raise ValueError("Only triangular elements are supported")
        
    def construct_initial_system(self):
        """
        Constructs the initial system of equations, saves the coefficient matrix in A
             and the right-hand side vector in b.
        """
        
        for element in self.EtoV:
            for r in range(3):
                for s in range(3):
                    self.A[element[r], element[s]] += self.compute_k(element, r, s)

    ##################### Boundary Condition Methods #####################

    def impose_BC(self, BC_type: str, BC: int, BC_func: callable):
        """
        Impose boundary conditions on the system. And saves BC_func in the corresponding boundary condition attribute.

        Parameters
        ----------
        BC : int
            Boundary condition index.
        BC_type : str
            Type of boundary condition.
        BC_func : callable
            Boundary condition function.
        """
        if BC_type == "Dirichlet":
            self.impose_Dirichlet_BC(BC, BC_func)
        elif BC_type == "Neumann":
            self.impose_Neumann_BC(BC, BC_func)
        else:
            raise ValueError("That is not a valid boundary condition type, it should either be Dirichlet or Neumann")
        
    def impose_Dirichlet_BC(self, BC: int, BC_func: callable):
        """
        Impose Dirichlet boundary conditions on the system. And saves BC_func in the corresponding boundary condition attribute.

        Parameters
        ----------
        BC : int
            Boundary condition index.
        BC_func : callable
            Boundary condition function.
        """

        BC_nodes = self.boundary_indices[BC]

        for node in BC_nodes:
            self.A[node, :] = 0
            self.A[node, node] = 1
            self.b[node] = BC_func(*self.points[node])

        if BC == 0:
            self.f0 = BC_func
        elif BC == 1:
            self.f1 = BC_func
        elif BC == 2:
            self.f2 = BC_func
        elif BC == 3:
            self.f3 = BC_func
        else:
            raise ValueError("That is not a valid boundary condition, it should be between 0 and 3")
        
    def impose_Neumann_BC(self, BC: int, BC_func: callable):
        """
        Impose Neumann boundary conditions on the system. And saves BC_func in the corresponding boundary condition attribute.

        Parameters
        ----------
        BC : int
            Boundary condition index.
        BC_func : callable
            Boundary condition function.
        """

        ##### TO BE IMPLEMENTED #####

        if BC == 0:
            self.f0 = BC_func
        elif BC == 1:
            self.f1 = BC_func
        elif BC == 2:
            self.f2 = BC_func
        elif BC == 3:
            self.f3 = BC_func
        else:
            raise ValueError("That is not a valid boundary condition, it should be between 0 and 3")
    
    ##################### Solution Methods #####################
    def solve(self):
        """
        Solves the finite element problem.
        """
        self.u = np.linalg.solve(self.A, self.b)

    def plot_solution(self, figsize = (8, 8), title = "Potential Flow Solution"):
        """
        Plots the solution.
        """
        u_plot = self.u.reshape(int(np.sqrt(self.M)), int(np.sqrt(self.M)))

        plt.figure(figsize=figsize)
        plt.imshow(u_plot, cmap="viridis", origin="lower", 
                   extent=[np.min(self.points[:,0]), np.max(self.points[:,0]), np.min(self.points[:,1]), 
                           np.max(self.points[:,1])])
        plt.title(title)
        plt.colorbar()
        plt.show()