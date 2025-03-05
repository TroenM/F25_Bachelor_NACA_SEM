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


def plot_mesh(mesh: meshio.Mesh,xlim : list = [-3,3], ylim : list = [-2,2], legend : bool=False, dpi=300) -> None:
    # Extract points and cells
    points = mesh.points[:, :2]  # Only take x, y for 2D
    cell_dict = mesh.cells_dict.keys()

    # Calculating x and y max and min
    x_min = np.min(points[:,0])
    x_max = np.max(points[:,0])
    xrange = x_max - x_min
    y_min = np.min(points[:,1])
    y_max = np.max(points[:,1])

    # setting legends, colors and linewidth
    BC_dict = {"in":1, "out":2, "deck":3, "fs":4, "naca":5}
    BC_colors = {"in":"darkgreen", "out":"darkred", "deck":"darkblue", "fs":"yellow", "naca":"red"}
    linew = 1.5

    # setting color and size for points
    point_color = "black"
    point_size = 1

    # Set y and x lim
    yrange = y_max - y_min
    margin = 0.03
    if xlim == [-3,3]:
        xlim = [x_min-margin*xrange, x_max+margin*xrange]
    if ylim == [-2,2]:
        ylim = [y_min-margin*yrange, y_max+margin*yrange]
    xrange = xlim[1] - xlim[0]
    yrange = ylim[1] - ylim[0]
    ranges = np.array([xrange,yrange])
    a = 20/xrange ; b = 20/yrange
    ranges *= min(a,b)
    fig, ax = plt.subplots(figsize=ranges, dpi=dpi)
    
    for celltype in cell_dict:
        if celltype in ["quad","triangle", "polygon"]:
            cells = mesh.cells_dict[celltype]  # Extract cells
            # Extract actual coordinates for plotting
            if len(cells.shape) > 1:
                cell_coords = [[points[point] for point in cell] for cell in cells]
                pc = PolyCollection(cell_coords, edgecolor="black", facecolor="lightblue", alpha=0.5)
            elif len(cells.shape) == 1:
                cell_coords = [[points[point] for point in cells]]
                pc = PolyCollection(cell_coords, edgecolor="black", facecolor="lightblue", alpha=0.5)
        if celltype == "line":
            lines = mesh.cells_dict[celltype]
            line_clasifications = mesh.cell_data_dict["gmsh:physical"]["line"]
            in_lines = lines[np.where(line_clasifications == BC_dict["in"])[0]]
            out_lines = lines[np.where(line_clasifications == BC_dict["out"])[0]]
            deck_lines = lines[np.where(line_clasifications == BC_dict["deck"])[0]]
            fs_lines = lines[np.where(line_clasifications == BC_dict["fs"])[0]]
            naca_lines = lines[np.where(line_clasifications == BC_dict["naca"])[0]]
            for i in range(len(in_lines)):
                line = points[in_lines[i], :]
                x_vals = line[:,0]
                y_vals = line[:,1]
                if i == 0:
                    ax.plot(x_vals,y_vals, linewidth = linew, color=BC_colors["in"], label=f"{BC_dict["in"]}")
                else:
                    ax.plot(x_vals,y_vals, linewidth = linew, color=BC_colors["in"])
            for i in range(len(out_lines)):
                line = points[out_lines[i], :]
                x_vals = line[:,0]
                y_vals = line[:,1]
                if i == 0:
                    ax.plot(x_vals,y_vals, linewidth = linew, color=BC_colors["out"], label=f"{BC_dict["out"]}")
                else:
                    ax.plot(x_vals,y_vals, linewidth = linew, color=BC_colors["out"])
            for i in range(len(deck_lines)):
                line = points[deck_lines[i], :]
                x_vals = line[:,0]
                y_vals = line[:,1]
                if i == 0:
                    ax.plot(x_vals,y_vals, linewidth = linew, color=BC_colors["deck"], label=f"{BC_dict["deck"]}")
                else:
                    ax.plot(x_vals,y_vals, linewidth = linew, color=BC_colors["deck"])
            for i in range(len(fs_lines)):
                line = points[fs_lines[i], :]
                x_vals = line[:,0]
                y_vals = line[:,1]
                if i == 0:
                    ax.plot(x_vals,y_vals, linewidth = linew, color=BC_colors["fs"], label=f"{BC_dict["fs"]}")
                else:
                    ax.plot(x_vals,y_vals, linewidth = linew, color=BC_colors["fs"])
            for i in range(len(naca_lines)):
                line = points[naca_lines[i], :]
                x_vals = line[:,0]
                y_vals = line[:,1]
                if i == 0:
                    ax.plot(x_vals,y_vals, linewidth = linew, color=BC_colors["naca"], label=f"{BC_dict["naca"]}")
                else:
                    ax.plot(x_vals,y_vals, linewidth = linew, color=BC_colors["naca"])
    

    
    ax.add_collection(pc)
    ax.scatter(points[:, 0], points[:, 1], color=point_color, s=point_size)  # Plot nodes
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    if legend:
        ax.legend()
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("2D Mesh Visualization")
    plt.show()
    return None



def find_point_on_boundary(point : np.ndarray, rot_mat : np.ndarray, inv_rot_mat : np.ndarray, func : callable = lambda x: 3*x/4) -> np.ndarray:
    vec = np.array([0,0])
    un_rotated_point = point @ inv_rot_mat
    BP = np.array([func(un_rotated_point[0]),0])
    BP = BP@rot_mat
    point_vec = point - BP
    y_sign = np.sign(point[1] - BP[1])
    x_sign = np.sign(point[0] - BP[0])
    b = 0
    c = 0
    if point_vec[1] != 0:
        b = ((2*y_sign - point[1])/point_vec[1])
    if point_vec[0] != 0:
        c = ((3*x_sign - point[0])/point_vec[0])
    if b != 0 and abs(b*point_vec[0] + point[0]) < 3:
        vec = point + point_vec*b
    else:
        vec = point + point_vec*c

    if -3 > vec[0] or vec[0] > 3 or vec[1] < -2 or 2 < vec[1]:
        a = 2
    if np.linalg.norm(vec) == 0:
        raise KeyError("Du tog ikke højde for dette scenarie")
    return vec


def transfinite_line(N : int, p1 : np.ndarray, p2 : np.ndarray, func : callable = lambda x: x**2) -> np.ndarray:
    """
    Generates a line from p1 to p2 with N points, where the spacing between points
    is dictated by the function `func`.
    
    Parameters:
        N (int): Number of points including p1 and p2.
        p1 (np.ndarray): Start point in the plane (2D or 3D).
        p2 (np.ndarray): End point in the plane (2D or 3D).
        func (callable): Function that defines the spacing transformation. 
                         It should take a parameter in [0, 1] and return a value in [0, 1].

    Returns:
        np.ndarray: An array of shape (N, dim) containing the generated points.
    """
    p1, p2 = np.array(p1), np.array(p2)
    assert p1.shape == p2.shape, "p1 and p2 must have the same dimensions"
    
    # Generate uniform parameter values in [0,1]
    s_uniform = np.linspace(0, 1, N+1)
    
    # Apply transformation function
    s_transformed = np.array([func(s) for s in s_uniform])
    
    # Normalize transformed values to stay in [0,1] range
    s_transformed = (s_transformed - s_transformed[0]) / (s_transformed[-1] - s_transformed[0])
    
    # Compute interpolated points
    points = (1 - s_transformed)[:, None] * p1 + s_transformed[:, None] * p2
    
    return points[1:,:]



def NACA_mesh(POA : int, PTA : int, NACA_name : str, gridtype : str = "quad", angle_of_attack : float = 0, func : callable = lambda x : x**2) -> meshio.Mesh:
    try:
        NACA_points = np.loadtxt(f"NACA_{NACA_name}.txt")
    except:
        raise ValueError("The data for this NACA-airfoil is not in the directory")
    # Defining points
    NACA_points = NACA_points[::(200//POA),:]
    NACA_points = NACA_points - np.array([0.5,0])
    POA = NACA_points.shape[0]
    angle_of_attack_rad = angle_of_attack*np.pi/180
    rot_mat = np.array([
        [np.cos(angle_of_attack_rad), -np.sin(angle_of_attack_rad)],
        [np.sin(angle_of_attack_rad), np.cos(angle_of_attack_rad)]
    ])
    inv_rot_mat = np.linalg.inv(rot_mat)
    NACA_points = (NACA_points)@rot_mat
    p2s = []
    for i in range(len(NACA_points)):
        p1 = NACA_points[i]
        p2s.append(find_point_on_boundary(p1,rot_mat, inv_rot_mat))
    p2s = np.array(p2s)
    for cornerpoint in [[-3,-2],[3,-2],[3,2],[-3,2]]:
        distances = np.linalg.norm(p2s - cornerpoint, axis=1)

        closest_index = np.argmin(distances)
        p2s[closest_index] = cornerpoint

    for i in range(len(NACA_points)):
        p1 = NACA_points[i]
        p2 = p2s[i]
        NACA_points = np.vstack([NACA_points,transfinite_line(PTA, p1, p2, func)])

    final_points = np.append(NACA_points, np.zeros((NACA_points.shape[0],1)), axis=1)
    # Defining cells
    if gridtype.lower() == "quad":
        amount_of_areas = (POA)*(PTA)
        areas = np.zeros((amount_of_areas,4), dtype=int)
        for j in range(POA):
            for i in range(PTA-1):
                areas[j+i*POA,:] = POA + ((np.array([j*PTA+i, j*PTA+(i+1), (j+1)*PTA+(i+1), (j+1)*PTA+i], dtype=int)) % ((POA)*(PTA+1)-POA))
        for j in range(POA):
            areas[POA*(PTA-1)+ j] = np.array([j, POA + j*PTA, POA + ((j+1)*PTA)%((POA)*(PTA+1)-POA), (j+1)%POA])
    if gridtype.lower() == "triangle":
        amount_of_areas = (POA)*(PTA)*2
        areas = np.zeros((amount_of_areas,3), dtype=int)
        for j in range(POA):
            for i in range(PTA-1):
                areas[(j+i*POA)*2,:] = POA + ((np.array([j*PTA+i, j*PTA+(i+1), (j+1)*PTA+(i+1)], dtype=int)) % ((POA)*(PTA+1)-POA))
                areas[(j+i*POA)*2+1,:] = POA + ((np.array([j*PTA+i, (j+1)*PTA+(i+1), (j+1)*PTA+i], dtype=int)) % ((POA)*(PTA+1)-POA))
        for j in range(POA):
            areas[(POA*(PTA-1)+ j)*2] = np.array([j, POA + j*PTA, POA + ((j+1)*PTA)%((POA)*(PTA+1)-POA)])
            areas[(POA*(PTA-1)+ j)*2+1] = np.array([j, POA + ((j+1)*PTA)%((POA)*(PTA+1)-POA), (j+1)%POA])
    
    # Defining boundaries
    ones = np.where(np.round(NACA_points[:,0],3) == -3)[0]
    nr_ones = len(ones)
    twos = np.where(np.round(NACA_points[:,0],3) == 3)[0]
    nr_twos = len(twos)
    threes = np.where(np.round(NACA_points[:,1],3) == -2)[0]
    nr_threes = len(threes)
    fours = np.where(np.round(NACA_points[:,1],3) == 2)[0]
    nr_fours = len(fours)
    fives = np.arange(POA)
    nr_fives = len(fives)
    lines = np.zeros((nr_fives + nr_ones + nr_twos + nr_threes + nr_fours - 4,2), dtype=int)
    for i in range(nr_ones-1):
        lines[i] = np.array([ones[i],ones[i+1]], dtype=int)
    for i in range(nr_twos-1):
        if abs(twos[i]-twos[i+1]) == PTA:
            lines[nr_ones-1 + i] = np.array([twos[i],twos[i+1]], dtype=int)
        else:
            lines[nr_ones-1 + i] = np.array([np.max(twos),np.min(twos)], dtype=int)
    for i in range(nr_threes-1):
        lines[nr_ones + nr_twos - 2 + i] = np.array([threes[i],threes[i+1]], dtype=int)
    for i in range(nr_fours-1):
        lines[nr_ones + nr_twos + nr_threes - 3 + i] = np.array([fours[i],fours[i+1]], dtype=int)
    for i in range(nr_fives):
        lines[nr_ones + nr_twos + nr_threes + nr_fours - 4 + i] = np.array([fives[i],fives[(i+1)%nr_fives]], dtype=int)

    cells = [("line",lines),(gridtype, areas)]
    mesh = meshio.Mesh(points=NACA_points, cells=cells)
    mesh.cell_data["gmsh:physical"] = [
        np.hstack((np.repeat(1,nr_ones-1), np.repeat(2,nr_twos-1), np.repeat(3,nr_threes-1),np.repeat(4,nr_fours-1), np.repeat(5,POA))),
        np.repeat(6,amount_of_areas)
    ]
    mesh = add_dummy_geometrical_data(mesh)
    mesh.points = np.array(mesh.points, dtype=np.float64)
    return mesh



def shift_surface(mesh : meshio.Mesh, func_before : callable, func_after : callable) -> meshio.Mesh:
    line_clasifications = mesh.cell_data_dict["gmsh:physical"]["line"]
    airfoil_lines = mesh.cells_dict["line"][np.where(line_clasifications == 5)[0]]
    airfoil_points = np.unique(airfoil_lines)

    
    point = mesh.points.copy()
    func_before = np.vectorize(func_before)
    func_after = np.vectorize(func_after)
    airfoil_values = point[airfoil_points]

    # Mask all points above airfoil
    point_mask = np.where((point[:,1] > np.max(airfoil_values[:,1])))
    min_point_val = np.min(point[point_mask,1])
    
    point[point_mask,1] -= min_point_val
    point[point_mask,1] = point[point_mask,1] * (func_after(point[point_mask,0]) /  func_before(point[point_mask,0]))
    point[point_mask,1] += min_point_val

    copy_mesh = mesh.copy()
    copy_mesh.points = point
    return copy_mesh


def naca_4digit(string : str, n : int) -> np.ndarray:
    """
    Returns the airfoil camber line and thickness distribution.
    Parameters
    ----------
    string : str
        A string consisting of 4 integers like "0012". This is the "name" of the airfoil
    n : int
        The number of points you want to modulate the airfoil with
    """
    if len(string) != 4:
        raise IndexError("The string needs to have the length 4")
    m = int(string[0])/100
    p = int(string[1])/10 if string[1] != "0" else 0.1
    t = int(string[2] + string[3])/100
    beta = np.linspace(0,np.pi,(n//2)+1)
    x = (1-np.cos(beta))/2
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    yc = np.where(x < p, m/p**2 * (2*p*x - x**2), m/(1-p)**2 * ((1-2*p) + 2*p*x - x**2))
    lower = np.hstack((x.reshape(-1,1), (yc - yt).reshape(-1,1)))[::-1]
    if (n//2)*2 != n:
        upper = ((np.hstack((x.reshape(-1,1), (yc + yt).reshape(-1,1)))))[1:]
    else:
        beta = np.linspace(0,np.pi,(n//2))
        x = (1-np.cos(beta))/2
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
        yc = np.where(x < p, m/p**2 * (2*p*x - x**2), m/(1-p)**2 * ((1-2*p) + 2*p*x - x**2))
        upper = ((np.hstack((x.reshape(-1,1), (yc + yt).reshape(-1,1)))))[1:]
    points = np.vstack((lower, upper))
    return points