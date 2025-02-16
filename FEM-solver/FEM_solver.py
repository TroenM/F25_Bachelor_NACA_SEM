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
        Impose boundary conditions on the system using impose_Dirichlet_BC or impose_Neumann_BC.

    impose_Dirichlet_BC(BC : int, BC_func : callable)
        Impose Dirichlet boundary conditions on the system. 
    
    impose_Neumann_BC(BC : int, BC_func : callable)
        Impose Neumann boundary conditions on the system. 
        
    Solution Methods
    -------
    solve()
        Solves the finite element problem.

    """


    ##################### ATTRIBUTES #####################
    mesh: meshio.Mesh
    EtoV: np.ndarray
    coords: np.ndarray
    boundary_indices: np.ndarray

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
        self.EtoV, self.coords = self.get_mesh_info(mesh)
        self.coords = self.coords[:, :2]

        self.N = len(self.EtoV)
        self.M = self.coords.shape[0]

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
        coords : np.ndarray
            3D array containing the point coordinates.

            Format:
            [x, y, z] for point 0
            [x, y, z] for point 1
            ...
        """

        # initialize list capable of holding multiple cell types
        EtoV = []

        # Try to extract triangular cells
        try :
            for element in mesh.cells_dict["triangle"]:
                EtoV.append(element)
        except:
            print("No triangle cells found")

        # Try to extract quad cells
        try :
            EtoV.append(mesh.cells_dict["quad"])
        except:
            print("No quad cells found")

        # Extract the node coordinates
        coords = mesh.points[:, :2]

        return EtoV, coords
    

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
        coords = self.coords[EtoV]

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

        ### TO DO: Implement for quad elements ###

        # Call the correct method based on the element type 
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
            for r in range(len(element)):
                for s in range(len(element)):
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
        BC_nodes = np.unique(self.mesh.cells_dict["line"][np.where(self.mesh.cell_data["gmsh:physical"][0] == BC)[0]].flatten())

        if len(BC_nodes) == 0:
            raise ValueError("No nodes found at boundary tag {}".format(BC))
        
        for node in BC_nodes:
            self.A[node, :] = 0
            self.A[node, node] = 1
            self.b[node] = BC_func(*self.coords[node])
        
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
        print("Not implemented yet")
    
    ##################### Solution Methods #####################
    def solve(self):
        """
        Solves the finite element problem.
        """
        #### MODIFY TO ACCEPT NON-SQUARE MESHES ####
        try:
            self.u = np.linalg.solve(self.A, self.b)
        except:
            raise ValueError("Plotting method is not implemented for non-square meshes")

    def plot_solution(self, figsize = (8, 8), title = "Potential Flow Solution"):
        """
        Plots the solution.
        """
        u_plot = self.u.reshape(int(np.sqrt(self.M)), int(np.sqrt(self.M)))

        plt.figure(figsize=figsize)
        plt.imshow(u_plot, cmap="viridis", origin="lower", 
                   extent=[np.min(self.coords[:,0]), np.max(self.coords[:,0]), np.min(self.coords[:,1]), 
                           np.max(self.coords[:,1])])
        plt.title(title)
        plt.colorbar()
        plt.show()