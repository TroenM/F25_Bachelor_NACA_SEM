import numpy as np
import scipy.sparse as sps
from scipy import integrate

import meshio

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.tri as tri

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
    v: np.ndarray
        The velocity field.
    

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
    
    Post-Processing Methods
    -------
    plot_solution(figsize : tuple, title : str)
        Plots the solution. Currently only supports square meshes.
    
    compute_velocity_field():
        Computes the velocity field v = gradient(u).
    
    """


    ##################### ATTRIBUTES #####################
    mesh: meshio.Mesh
    EtoV: np.ndarray
    coords: np.ndarray

    N: int
    M: int
    A: sps.csr_matrix
    b: np.ndarray
    sol: np.ndarray
    v: np.ndarray
    rhs: callable


    ##################### INITIALIZATION #####################
    def __init__(self, mesh: meshio.Mesh, rhs: callable = lambda x, y: 0):
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
        self.sol = np.zeros(self.M)
        self.v = np.zeros((self.M, 2))
        self.rhs = rhs

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

        return 1/(4*np.abs(Delta)) * (br*bs + cr*cs)
    
    def compute_quad_k(self, EtoV: np.ndarray, r, s) -> float:
        """
        Computes the element in element matrix for a quad element.
        """

        # Get the coordinates of the vertices
        coords = self.coords[EtoV]

        # Declaring x_i, y_i for i = 0, 1, 2, 3
        x = coords[:, 0]
        y = coords[:, 1]

        # Compute the area of the quad element
    
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
        
    def compute_rhs(self, element: np.ndarray, r) -> float:
        """
        Computes the Right hand side for a given element
        
        Parameters
        ----------
        element: EtoV infor for the element
        r: row index


        """

        ### Parameterizing the element ###
        coords = self.coords[element]

        x1,y1 = coords[0]
        x2,y2 = coords[1]
        x3,y3 = coords[2]

        #x = (x1+x2+x3)/3
        #y = (y1+y2+y3)/3

        q = (self.rhs(x1,y1) + self.rhs(x2,y2) + self.rhs(x3,y3))/3

        # Constructing coefficients of the basis functions
        #ar = coords[(r+1)%3, 0]*coords[(r+2)%3, 1] - coords[(r+2)%3, 0]*coords[(r+1)%3, 1]  # xj*yk - xk*yj
        #br = coords[(r+1)%3,1] - coords[(r+2)%3,1]
        #cr = coords[(r+2)%3,0] - coords[(r+1)%3,0]

        Delta = 1/2 * (coords[1, 0]*coords[2, 1] - coords[1,1]*coords[2, 0]
                           -(coords[0, 0]*coords[2, 1] - coords[0,1]*coords[2, 0])
                           + coords[0, 0]*coords[1, 1] - coords[0,1]*coords[1, 0])

        qr = abs(Delta)/3 * q

        return qr
        

    def construct_initial_system(self):
        """
        Constructs the initial system of equations, saves the coefficient matrix in A
             and the right-hand side vector in b.
        """
        
        for element in self.EtoV:
            for r in range(len(element)):
                self.b[element[r]] += self.compute_rhs(element, r)
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
        if BC_type.upper() == "DIRICHLET":
            self.impose_Dirichlet_BC(BC, BC_func)
        elif BC_type.upper() == "NEUMANN":
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

        self.sol = np.linalg.solve(self.A, self.b)

    def compute_velocity_field(self):
        """
        Computes the velocity field v = gradient(u) using 1.order downwind differences.
        """
        # Converting the solution to a 2D array
        self.sol = self.sol.reshape(int(np.sqrt(self.M)), int(np.sqrt(self.M)))
        self.v = self.v.reshape(int(np.sqrt(self.M)), int(np.sqrt(self.M)), 2)

        # Computing x-distances between nodes for downwind differences
        coords_x = self.coords[:, 0].reshape(int(np.sqrt(self.M)), int(np.sqrt(self.M)))
        dx = coords_x[:, 1:] - coords_x[:, :-1]

        # Computing y-distances between nodes for downwind differences
        coords_y = self.coords[:, 1].reshape(int(np.sqrt(self.M)), int(np.sqrt(self.M)))
        dy = coords_y[1:, :] - coords_y[:-1, :]

        # Computing the velocity field
        self.v[:,:-1, 0] = np.diff(self.sol, axis=1) / dx
        self.v[:-1,:, 1] = np.diff(self.sol, axis=0) / dy

        # Flattening the velocity field
        self.v = self.v.reshape(self.M, 2)

        # restoring the solution to a 1D array
        self.sol = self.sol.flatten()

    ##################### Post-Processing Methods #####################
    def plot_solution(self, show_elements: bool = True, figsize: tuple = (10, 10), title: str = "Potential Flow Solution"):
        """
        Plots the solution.

        Inspired by: https://stackoverflow.com/questions/52202014/how-can-i-plot-2d-fem-results-using-matplotlib
        Answer by Carlos Jan 29 2020

        Parameters
        ----------
        figsize : tuple
            Figure size.
        title : str
            Title of the plot.
        """
        triangulation = tri.Triangulation(self.coords[:,0], self.coords[:, 1], self.EtoV )
        plt.tricontourf(triangulation, self.sol, cmap="plasma", levels = 100)

        if show_elements:
            for element in self.EtoV:
                x = [self.coords[:, 0][element[i]] for i in range(len(element))]
                y = [self.coords[:, 1][element[i]] for i in range(len(element))]
                plt.fill(x, y, edgecolor='black', fill=False, linewidth=0.5)
        
        plt.colorbar()
        plt.title(title)
        plt.axis("equal")
        plt.show()

    def plot3d(self, show_elements: bool = True, cmap: str = "thermal"):
        """
        Plot the solution in 3D as nodes and elements
        """

        points_3d = np.hstack((self.coords, self.sol.reshape(-1, 1)))
        EtoV = np.array(self.EtoV)
        mesh_3d = go.Mesh3d(
                x = points_3d[:, 0],
                y = points_3d[:, 1],
                z = points_3d[:, 2],
                i = EtoV[:, 0],
                j = EtoV[:, 1],
                k = EtoV[:, 2],
                opacity = 1,
                colorscale = cmap,
                intensity = points_3d[:, 2],
                colorbar_title = "Stream function"
            )
        
        if show_elements:

            tri_points = points_3d[self.EtoV]
            #extract the lists of x, y, z coordinates of the triangle vertices and connect them by a line
            Xe = []
            Ye = []
            Ze = []
            for T in tri_points:
                Xe.extend([T[k%3][0] for k in range(4)]+[ None])
                Ye.extend([T[k%3][1] for k in range(4)]+[ None])
                Ze.extend([T[k%3][2] for k in range(4)]+[ None])

            #define the trace for triangle sides
            lines = go.Scatter3d(
                               x=Xe,
                               y=Ye,
                               z=Ze,
                               mode='lines',
                               name='',
                               opacity=1,
                               line=dict(color= 'rgb(10,10,10)', width=0.5)) 


            fig = go.Figure(data=[mesh_3d, lines])
        else:
            fig = go.Figure(data=[mesh_3d])
    
        fig.show()


if __name__ == "__main__":
    print("Reading mesh\n")
    mesh = meshio.read("./FEM-solver/meshes/NACAmesh.msh")
    print("Initializing solver")
    model = PotentialFlowSolver_FEM(mesh)
    print("Imposing BCs\n")
    model.impose_BC("Dirichlet", 1, lambda x, y: 10)
    model.impose_BC("Dirichlet", 2, lambda x, y: 10)
    print("Solving\n")
    model.solve()
    print("Plotting\n")
    model.plot_solution(show_elements=False, figsize=(7,7))