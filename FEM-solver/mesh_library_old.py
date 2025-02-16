import numpy as np
import meshio
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection



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
    first_list = np.tile(np.linspace(-2,3,N_cols),N_rows)
    second_list = np.repeat(np.linspace(-2,2,N_rows),N_cols)
    points = np.array(((first_list,second_list,np.zeros(N_cols*N_rows)))).T
    boundary_edges = np.array([
            np.where((points[:, 1] == np.min(points[:, 1])))[0],   # Left boundary
            np.where((points[:, 1] == np.max(points[:, 1])))[0],   # Right boundary
            np.where((points[:, 0] == np.min(points[:, 0])))[0],   # Bottom boundary
            np.where((points[:, 0] == np.max(points[:, 0])))[0]    # Top boundary
    ])
    boundary_markers = {"bottom": [0], "right": [1], "top": [2], "left": [3]} 
    if gridtype.lower() == "quad":
        squares = np.zeros(((N_cols-1)*(N_rows-1),4), dtype=int)
        for i in range(N_cols-1):
            for j in range(N_rows-1):
                squares[(N_cols-1)*j+i,:] = [i+j*N_cols, i+1+j*N_cols, i+1+(j+1)*N_cols, i+(j+1)*N_cols]
        cells = [(gridtype, squares)]
        return meshio.Mesh(points=points, cells=cells, cell_sets={"boundary": boundary_edges})
    
    elif gridtype.lower() == "triangle":
        triangles = np.zeros(((N_cols-1)*(N_rows-1)*2,3), dtype=int)
        for i in range(N_cols-1):
            for j in range(N_rows-1):
                triangles[((N_cols-1)*j+i)*2,:] = [i+j*N_cols, i+1+j*N_cols, i+1+(j+1)*N_cols]
                triangles[((N_cols-1)*j+i)*2+1,:] = [i+j*N_cols, i+1+(j+1)*N_cols, i+(j+1)*N_cols]
        cells = [(gridtype, triangles)]
        return meshio.Mesh(points=points, cells=cells, cell_sets={"boundary": boundary_edges})
    else:
        raise ValueError("That is not a valid gridtype, it should either be triangle or quad")


def plot_mesh(mesh: meshio.Mesh) -> None:
    # Extract points and cells
    points = mesh.points[:, :2]  # Only take x, y for 2D
    gridtype = list(mesh.cells_dict.keys())[0]
    cells = mesh.cells_dict[gridtype]  # Extract cells
    fig, ax = plt.subplots(figsize=(6, 6))
    # Extract actual coordinates for plotting
    if len(cells.shape) > 1:
        cell_coords = [[points[point] for point in cell] for cell in cells]
        pc = PolyCollection(cell_coords, edgecolor="black", facecolor="lightblue", alpha=0.5)
    elif len(cells.shape) == 1:
        cell_coords = [[points[point] for point in cells]]
        pc = PolyCollection(cell_coords, edgecolor="black", facecolor="lightblue", alpha=0.5)

    
    ax.add_collection(pc)
    ax.scatter(points[:, 0], points[:, 1], color="red", s=10)  # Plot nodes
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("2D Mesh Visualization")
    plt.axis("equal")
    plt.show()
    return None