import numpy as np
import meshio
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

def add_dummy_geometrical_data(mesh) -> meshio.Mesh:
    
    # Ensure cell_data exists and has 'gmsh:geometrical'
    if "gmsh:geometrical" not in mesh.cell_data:
        mesh.cell_data["gmsh:geometrical"] = {}
    # Assign geometrical tags
    mesh.cell_data["gmsh:geometrical"] = []
    for i in range(len(mesh.cells_dict.keys())):
        mesh.cell_data["gmsh:geometrical"].append(np.zeros(len(mesh.cell_data["gmsh:physical"][i]), dtype=int))  # Default to entity 0
    return mesh


def mesh_gen_uniform_2D_grid(N_rows: int, N_cols: int,gridtype: str) -> meshio.Mesh:
    '''
    Parameters
    ---
    N_rows : int
        Number of rows in the uniform grid
    N_cols : int
        Number of collumns in the uniform grid
    gridtype : str
        The gridtype is "triangle" if you want the grid to be made up of triangles, or "quad" if you want your grid to be made of squares
    '''
    first_list = np.tile(np.linspace(-3,3,N_cols),N_rows)
    second_list = np.repeat(np.linspace(-2,2,N_rows),N_cols)
    points = np.array(((first_list,second_list,np.zeros(N_cols*N_rows)))).T
    if gridtype.lower() == "quad":
        amount_of_squares = (N_cols-1)*(N_rows-1)
        squares = np.zeros((amount_of_squares,4), dtype=int)
        for i in range(N_cols-1):
            for j in range(N_rows-1):
                squares[(N_cols-1)*j+i,:] = [i+j*N_cols, i+1+j*N_cols, i+1+(j+1)*N_cols, i+(j+1)*N_cols]
        
        lines = np.zeros((2*(N_cols-1) + 2*(N_rows-1),2), dtype=int)
        for j in range(2):
            for i in range(N_rows-1):
                lines[i+j*(N_rows-1),:] = np.array([i*N_cols,(i+1)*N_cols]) + j*(N_cols-1)
        for j in range(2):
            for i in range(N_cols-1):
                lines[2*(N_rows-1)+i+j*(N_cols-1),:] = np.array([i,i+1]) + j*(N_cols)*(N_rows-1)
        cells = [("line",lines),(gridtype, squares)]
        mesh = meshio.Mesh(points=points, cells=cells)
        mesh.cell_data["gmsh:physical"] = [
            np.hstack((np.repeat(1,N_rows-1), np.repeat(2,N_rows-1), np.repeat(3,N_cols-1),np.repeat(4,N_cols-1))),
            np.repeat(6,amount_of_squares)
        ]
        mesh = add_dummy_geometrical_data(mesh)
        return mesh
        
    
    elif gridtype.lower() == "triangle":
        amount_of_triangles = (N_cols-1)*(N_rows-1)*2
        triangles = np.zeros((amount_of_triangles,3), dtype=int)
        for i in range(N_cols-1):
            for j in range(N_rows-1):
                triangles[((N_cols-1)*j+i)*2,:] = [i+j*N_cols, i+1+j*N_cols, i+1+(j+1)*N_cols]
                triangles[((N_cols-1)*j+i)*2+1,:] = [i+j*N_cols, i+1+(j+1)*N_cols, i+(j+1)*N_cols]
        
        lines = np.zeros((2*(N_cols-1) + 2*(N_rows-1),2), dtype=int)
        for j in range(2):
            for i in range(N_rows-1):
                lines[i+j*(N_rows-1),:] = np.array([i*N_cols,(i+1)*N_cols]) + j*(N_cols-1)
        for j in range(2):
            for i in range(N_cols-1):
                lines[2*(N_rows-1)+i+j*(N_cols-1),:] = np.array([i,i+1]) + j*(N_cols)*(N_rows-1)
        cells = [("line",lines), (gridtype, triangles)]
        mesh = meshio.Mesh(points=points, cells=cells)
        mesh.cell_data["gmsh:physical"] = [
            np.hstack((np.repeat(1,N_rows-1), np.repeat(2,N_rows-1), np.repeat(3,N_cols-1),np.repeat(4,N_cols-1))),
            np.repeat(6,amount_of_triangles)
        ]
        mesh = add_dummy_geometrical_data(mesh)
        return mesh
    else:
        raise ValueError("That is not a valid gridtype, it should either be triangle or quad")


def plot_mesh(mesh: meshio.Mesh, celltype : str = "quad",xlim : list = [-3,3], ylim : list = [-2,2]) -> None:
    # Extract points and cells
    points = mesh.points[:, :2]  # Only take x, y for 2D
    cell_dict = mesh.cell_data_dict.keys()
    for celltype in cell_dict:
        cells = mesh.cells_dict[celltype]  # Extract cells
        # Extract actual coordinates for plotting
        if len(cells.shape) > 1:
            cell_coords = [[points[point] for point in cell] for cell in cells]
            pc = PolyCollection(cell_coords, edgecolor="black", facecolor="lightblue", alpha=0.5)
        elif len(cells.shape) == 1:
            cell_coords = [[points[point] for point in cells]]
            pc = PolyCollection(cell_coords, edgecolor="black", facecolor="lightblue", alpha=0.5)
    

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_collection(pc)
    ax.scatter(points[:, 0], points[:, 1], color="red", s=1)  # Plot nodes
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("2D Mesh Visualization")
    plt.show()
    return None