import numpy as np
import meshio
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import os
from scipy.interpolate import interp1d

try:
    import gmsh
except:
    print("Could not import gmsh")

try:
    import firedrake as fd
except:
    print("Could not import firedrake")

def add_dummy_geometrical_data(mesh) -> meshio.Mesh:
    
    # Ensure cell_data exists and has 'gmsh:geometrical'
    if "gmsh:geometrical" not in mesh.cell_data:
        mesh.cell_data["gmsh:geometrical"] = {}
    # Assign geometrical tags
    mesh.cell_data["gmsh:geometrical"] = []
    for i in range(len(mesh.cells_dict.keys())):
        mesh.cell_data["gmsh:geometrical"].append(np.zeros(len(mesh.cell_data["gmsh:physical"][i]), dtype=int))  # Default to entity 0
    return mesh


def mesh_gen_uniform_2D_grid(N_rows: int, N_cols: int,gridtype: str, xlim: list = [-3,3], ylim: list = [-2,2]) -> meshio.Mesh:
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
    # Repeat a list [1,2,3] in the xlim range, N_row times. example [1,2,3,1,2,3,1,2,3]
    first_list = np.tile(np.linspace(xlim[0], xlim[1],N_cols),N_rows)

    # Repeat entries of a list [1,2,3] in the xlim range, N_row times. example: [1,1,1,2,2,2,3,3,3]
    second_list = np.repeat(np.linspace(ylim[0], ylim[1],N_rows),N_cols)

    # Define point coordinates
    points = np.array(((first_list,second_list,np.zeros(N_cols*N_rows)))).T
    
    # If grid should be made of squares
    if gridtype.lower() == "quad":

        # Calculate amount of squares and an array to hold the information
        amount_of_squares = (N_cols-1)*(N_rows-1)
        squares = np.zeros((amount_of_squares,4), dtype=int)

        # Create the squares from the indecies of their cornerpoints
        for i in range(N_cols-1):
            for j in range(N_rows-1):
                squares[(N_cols-1)*j+i,:] = [i+j*N_cols, i+1+j*N_cols, i+1+(j+1)*N_cols, i+(j+1)*N_cols]
        
        # Create lines for the boundaries
        lines = np.zeros((2*(N_cols-1) + 2*(N_rows-1),2), dtype=int)
        for j in range(2):
            for i in range(N_rows-1):
                lines[i+j*(N_rows-1),:] = np.array([i*N_cols,(i+1)*N_cols]) + j*(N_cols-1)
        for j in range(2):
            for i in range(N_cols-1):
                lines[2*(N_rows-1)+i+j*(N_cols-1),:] = np.array([i,i+1]) + j*(N_cols)*(N_rows-1)

        # Save the lines and squares such that meshio can save it
        cells = [("line",lines),(gridtype, squares)]
        mesh = meshio.Mesh(points=points, cells=cells)
        mesh.cell_data["gmsh:physical"] = [
            np.hstack((np.repeat(1,N_rows-1), np.repeat(2,N_rows-1), np.repeat(3,N_cols-1),np.repeat(4,N_cols-1))),
            np.repeat(6,amount_of_squares)
        ]

        # Add irrelevant information such that meshio mesh can be converted into a firedrake mesh
        mesh = add_dummy_geometrical_data(mesh)
        return mesh
        
    # If the mesh should be made of triangles
    elif gridtype.lower() == "triangle":

        # Calculate amount of triangles, and initialize array to hold information
        amount_of_triangles = (N_cols-1)*(N_rows-1)*2
        triangles = np.zeros((amount_of_triangles,3), dtype=int)
        
        # Define triangles from the indicies of their cornerpoints
        for i in range(N_cols-1):
            for j in range(N_rows-1):
                triangles[((N_cols-1)*j+i)*2,:] = [i+j*N_cols, i+1+j*N_cols, i+1+(j+1)*N_cols]
                triangles[((N_cols-1)*j+i)*2+1,:] = [i+j*N_cols, i+1+(j+1)*N_cols, i+(j+1)*N_cols]
        
        # Define lines at the boundaries from the indecies of their cornerpoints
        lines = np.zeros((2*(N_cols-1) + 2*(N_rows-1),2), dtype=int)
        for j in range(2):
            for i in range(N_rows-1):
                lines[i+j*(N_rows-1),:] = np.array([i*N_cols,(i+1)*N_cols]) + j*(N_cols-1)
        for j in range(2):
            for i in range(N_cols-1):
                lines[2*(N_rows-1)+i+j*(N_cols-1),:] = np.array([i,i+1]) + j*(N_cols)*(N_rows-1)

        # Save information such that a meshio mesh can be created
        cells = [("line",lines), (gridtype, triangles)]
        mesh = meshio.Mesh(points=points, cells=cells)
        mesh.cell_data["gmsh:physical"] = [
            np.hstack((np.repeat(1,N_rows-1), np.repeat(2,N_rows-1), np.repeat(3,N_cols-1),np.repeat(4,N_cols-1))),
            np.repeat(6,amount_of_triangles)
        ]

        # Add irrelevant information such that meshio mesh can be converted into a firedrake mesh
        mesh = add_dummy_geometrical_data(mesh)
        return mesh
    
    # Return error if gridtype was neither quad or triangle
    else:
        raise ValueError("That is not a valid gridtype, it should either be triangle or quad")


def plot_mesh(mesh: meshio.Mesh,xlim : list = [-3,3], ylim : list = [-2,2], legend : bool=False, dpi=300, show=True, axes=False) -> None:
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
    linew = 0.8
    boundary_linew = 1.2

    # setting color and size for points
    point_color = "black"
    point_size = 0

    # Set y and x lim based on minimum and maximum x and y value
    yrange = y_max - y_min
    margin = 0.03
    if xlim == [-3,3]:
        xlim = [x_min-margin*xrange, x_max+margin*xrange]
    if ylim == [-2,2]:
        ylim = [y_min-margin*yrange, y_max+margin*yrange]
    xrange = xlim[1] - xlim[0]
    yrange = ylim[1] - ylim[0]

    # Set figsize
    ranges = np.array([xrange,yrange])
    a = 20/xrange ; b = 20/yrange
    ranges *= min(a,b)

    # Create the plot if plot mesh should not be a part of a subplot
    if axes:
        ax = axes
    else:
        fig, ax = plt.subplots(figsize=ranges, dpi=dpi)
    
    # Loop through different cell types
    for celltype in cell_dict:

        # If cell is an element
        if celltype in ["quad","triangle", "polygon"]:
            cells = mesh.cells_dict[celltype]  # Extract cells

            # Extract actual coordinates for plotting and plot them with a lightblue facecolor
            if len(cells.shape) > 1:
                cell_coords = [[points[point] for point in cell] for cell in cells]
                pc = PolyCollection(cell_coords, edgecolor="black", facecolor="lightblue", alpha=0.5)
            elif len(cells.shape) == 1:
                cell_coords = [[points[point] for point in cells]]
                pc = PolyCollection(cell_coords, edgecolor="black", facecolor="lightblue", alpha=0.5)

        # If cell is a line
        if celltype == "line":
            # Classify lines into different boundaries
            lines = mesh.cells_dict[celltype]
            line_clasifications = mesh.cell_data_dict["gmsh:physical"]["line"]
            in_lines = lines[np.where(line_clasifications == BC_dict["in"])[0]]
            out_lines = lines[np.where(line_clasifications == BC_dict["out"])[0]]
            deck_lines = lines[np.where(line_clasifications == BC_dict["deck"])[0]]
            fs_lines = lines[np.where(line_clasifications == BC_dict["fs"])[0]]
            naca_lines = lines[np.where(line_clasifications == BC_dict["naca"])[0]]

            # Plot lines on the inlet with the right color
            for i in range(len(in_lines)):
                line = points[in_lines[i], :]
                x_vals = line[:,0]
                y_vals = line[:,1]
                if i == 0:
                    ax.plot(x_vals,y_vals, linewidth = boundary_linew, color=BC_colors["in"], label=f"{BC_dict['in']}")
                else:
                    ax.plot(x_vals,y_vals, linewidth = boundary_linew, color=BC_colors["in"])

            # Plot lines on the outlet with the correct color
            for i in range(len(out_lines)):
                line = points[out_lines[i], :]
                x_vals = line[:,0]
                y_vals = line[:,1]
                if i == 0:
                    ax.plot(x_vals,y_vals, linewidth = boundary_linew, color=BC_colors["out"], label=f"{BC_dict['out']}")
                else:
                    ax.plot(x_vals,y_vals, linewidth = boundary_linew, color=BC_colors["out"])

            # Plot lines on the bed with the correct color
            for i in range(len(deck_lines)):
                line = points[deck_lines[i], :]
                x_vals = line[:,0]
                y_vals = line[:,1]
                if i == 0:
                    ax.plot(x_vals,y_vals, linewidth = boundary_linew, color=BC_colors["deck"], label=f"{BC_dict['deck']}")
                else:
                    ax.plot(x_vals,y_vals, linewidth = boundary_linew, color=BC_colors["deck"])

            # Plot lines on the free surface with correct color
            for i in range(len(fs_lines)):
                line = points[fs_lines[i], :]
                x_vals = line[:,0]
                y_vals = line[:,1]
                if i == 0:
                    ax.plot(x_vals,y_vals, linewidth = boundary_linew, color=BC_colors["fs"], label=f"{BC_dict['fs']}")
                else:
                    ax.plot(x_vals,y_vals, linewidth = boundary_linew, color=BC_colors["fs"])

            # Plot lines on the NACA-airfoil with the correct color
            for i in range(len(naca_lines)):
                line = points[naca_lines[i], :]
                x_vals = line[:,0]
                y_vals = line[:,1]
                if i == 0:
                    ax.plot(x_vals,y_vals, linewidth = boundary_linew, color=BC_colors["naca"], label=f"{BC_dict['naca']}")
                else:
                    ax.plot(x_vals,y_vals, linewidth = boundary_linew, color=BC_colors["naca"])
    

    # Add the elements to the plot
    ax.add_collection(pc)

    # Add nodes in the mesh
    ax.scatter(points[:, 0], points[:, 1], color=point_color, s=point_size)

    # Set x and y limits, and include legend if relevant
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    if legend:
        ax.legend()
    
    # Plot
    plt.xlabel("X")
    plt.ylabel("Y")
    if not axes:
        plt.title("2D Mesh Visualization")
    if show:
        plt.show()
    return None


def shift_surface(mesh : meshio.Mesh, func_b : callable, func_a : callable) -> meshio.Mesh:
    
    # Locate points a part of the airfoil
    line_clasifications = mesh.cell_data_dict["gmsh:physical"]["line"]
    airfoil_lines = mesh.cells_dict["line"][np.where(line_clasifications == 5)[0]]
    airfoil_points = np.unique(airfoil_lines)

    # Find concrete coordinates for the points on the airfoil
    point = mesh.points.copy()
    airfoil_values = point[airfoil_points]

    # Mask all points above the largest value of the airfoil
    point_mask = np.where((point[:,1] > np.max(airfoil_values[:,1])))
    
    # Define the largest y val of the airfoil as a new "minimum"
    min_point_val = np.max(airfoil_values[:,1])

    # Manipulate functions
    func_before = lambda x: func_b(x)-min_point_val
    func_after = lambda x: func_a(x)-min_point_val
    func_before = np.vectorize(func_before)
    func_after = np.vectorize(func_after)
    
    # Manipulate point values as stated in the report under methodology for free surface.
    point[point_mask,1] -= min_point_val
    point[point_mask,1] = point[point_mask,1] * (func_after(point[point_mask,0]) /  func_before(point[point_mask,0]))
    point[point_mask,1] += min_point_val

    # Copy the mesh and set new points to return a new mesh.
    copy_mesh = mesh.copy()
    copy_mesh.points = point
    return copy_mesh


def naca_4digit(string : str, n : int, alpha : float = 0, position_of_center : np.ndarray = np.array([0.5,0])) -> np.ndarray:
    """
    Returns the airfoil camber line and thickness distribution.
    Parameters
    ----------
    string : str
        A string consisting of 4 integers like "0012". This is the "name" of the airfoil
    n : int
        The number of points you want to modulate the airfoil with
    """

    # Raise error if name of NACA-airfoil is not 4 long
    if len(string) != 4:
        raise IndexError("The string needs to have the length 4")
    
    # Fetch information from the name of the airfoil
    m = int(string[0])/100
    p = int(string[1])/10 if string[1] != "0" else 0.1
    t = int(string[2] + string[3])/100

    # If there should be an equal amount of nodes on the airfoil
    if (n//2)*2 == n:

        # Create a linspace with points spaced equally between 0 and pi for the upper and lower surface of the airfoil
        beta = np.linspace(0,np.pi,(n//2)+1)

        # Convert this linspace to an array with points that are the most concentrated at the edges of the airfoil.
        x = (1-np.cos(beta))/2

        # Define the thickness of the airfoil distrebution in terms of x
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)

        # Define the camber in terms of x
        yc = np.where(x < p, m/p**2 * (2*p*x - x**2), m/(1-p)**2 * ((1-2*p) + 2*p*x - x**2))

        # Create the lower and upper part of the airfoil
        lower = np.hstack((x.reshape(-1,1), (yc - yt).reshape(-1,1)))[::-1][:-1]
        upper = np.hstack((x.reshape(-1,1), (yc + yt).reshape(-1,1)))[:-1]

    # If there should be an odd amount of nodes on the airfoil
    elif (n//2)*2 != n:

        # Create a linspace with points spaced equally between 0 and pi for the lower surface of the airfoil with one more point than for the upper part
        beta = np.linspace(0,np.pi,(n//2)+2)

        # Convert this linspace to an array with points that are the most concentrated at the edges of the airfoil.
        x = (1-np.cos(beta))/2
        
        # Define the thickness of the airfoil distrebution in terms of x
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)

        # Define the camber in terms of x
        yc = np.where(x < p, m/p**2 * (2*p*x - x**2), m/(1-p)**2 * ((1-2*p) + 2*p*x - x**2))

        # Define the lower part of the airfoil
        lower = np.hstack((x.reshape(-1,1), (yc - yt).reshape(-1,1)))[::-1][:-1]

        # Create a linspace with points spaced equally between 0 and pi for the upper surface of the airfoil
        beta = np.linspace(0,np.pi,(n//2)+1)

        # Convert this linspace to an array with points that are the most concentrated at the edges of the airfoil.
        x = (1-np.cos(beta))/2
        
        # Define the thickness of the airfoil distrebution in terms of x
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)

        # Define the camber in terms of x
        yc = np.where(x < p, m/p**2 * (2*p*x - x**2), m/(1-p)**2 * ((1-2*p) + 2*p*x - x**2))

        # Define the upper part of the airfoil
        upper = np.hstack((x.reshape(-1,1), (yc + yt).reshape(-1,1)))[:-1]
    
    # Stack the points of the lower surface on top of the points from the upper surface
    points = np.vstack((lower, upper))

    # Relocate the airfoil such that it haves its center at the origin
    points -= np.array([0.5,0])

    # Rotate the airfoil
    alpha = np.deg2rad(alpha)
    rot_matrix = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    points = np.dot(points, rot_matrix)

    # Relocate the airfoil such that it haves its center as desired
    points += position_of_center

    # Return points defining the airfoil
    return points


def naca_mesh(airfoil: str, alpha: float = 0, xlim: tuple = (-7,13), ylim: tuple = (-2,1), **kwargs):
    """
    
    parameters
    ----------
    airfoil: str
        directory of the airfoil coordinates in txt format
    alpha: float
        angle of attack in degrees
    xlim: tuple
        x limits of the domain
    ylim: tuple
        y limits of the domain
    
    ** kwargs:
        - n_in, n_out, n_bed, n_fs, n_airfoil: int
            Number of points in the inlet, outlet, bed, front step and airfoil
        - prog_in, prog_out, prog_bed, prog_fs: float
            Progression in the inlet, outlet, bed and front step
        - scale: float
            Element scale factor
        
    """

    # ==================== Initializing the model ====================
    try:
        kwargs = kwargs["kwargs"]
    except:
        pass
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)

    # Creating domain
    xmin, xmax = xlim
    ymin, ymax = ylim

    scale = kwargs.get('scale', 1.034)
    gmsh.model.geo.addPoint(xmin, ymin, 0, scale, tag=1) # bottom left
    gmsh.model.geo.addPoint(xmin, ymax, 0, scale, tag=2) # top left
    gmsh.model.geo.addPoint(xmax, ymin, 0, scale, tag=3) # bottom right
    gmsh.model.geo.addPoint(xmax, ymax, 0, scale, tag=4) # top right

    inlet = gmsh.model.geo.addLine(1, 2, tag=1)
    outlet = gmsh.model.geo.addLine(3, 4, tag=2)
    bed = gmsh.model.geo.addLine(1, 3, tag=3)
    fs = gmsh.model.geo.addLine(2, 4, tag=4)

    boundary_loop = gmsh.model.geo.addCurveLoop([inlet, fs, -outlet, -bed], tag=1)

    # ==================== Handling the airfoil ====================
    n_airfoil = kwargs.get('n_airfoil') if kwargs.get('n_airfoil') else 60
    coords = naca_4digit(airfoil, n = n_airfoil, alpha=alpha, position_of_center=kwargs.get("center_of_airfoil",np.array([0.5,0])))


    points = []
    for idx, coord in enumerate(coords[::len(coords)//n_airfoil]):
        points.append(gmsh.model.geo.addPoint(coord[0], coord[1], 0, scale, tag=5+idx))
    
    # Create lines
    lines = []
    for idx in range(len(points)-1):
        line = gmsh.model.geo.addLine(points[idx], points[idx+1], tag = 4+idx + 1)
        lines.append(line)
    
    
    line = gmsh.model.geo.addLine(points[-1], points[0], tag = 4+len(points))
    lines.append(line)

    airfoil_line = gmsh.model.geo.addCurveLoop(lines, tag=5)

    # ==================== Creating the surface ====================

    # Create the surface
    gmsh.model.geo.addPlaneSurface([boundary_loop, airfoil_line], tag=1)

    gmsh.model.geo.synchronize()

    # ==================== Physical groups ====================

    # Inlet
    gmsh.model.addPhysicalGroup(1, [inlet], tag=1)
    gmsh.model.setPhysicalName(1, 1, 'inlet')

    # Outlet
    gmsh.model.addPhysicalGroup(1, [outlet], tag=2)
    gmsh.model.setPhysicalName(1, 2, 'outlet')

    # Bed
    gmsh.model.addPhysicalGroup(1, [bed], tag=3)
    gmsh.model.setPhysicalName(1, 3, 'bed')

    # Free surface
    gmsh.model.addPhysicalGroup(1, [fs], tag=4)
    gmsh.model.setPhysicalName(1, 4, 'free_surface')

    # Airfoil
    gmsh.model.addPhysicalGroup(1, lines, tag=5)
    gmsh.model.setPhysicalName(1, 5, 'airfoil')

    # Domain
    gmsh.model.addPhysicalGroup(2, [1], tag=6)
    gmsh.model.setPhysicalName(2, 6, 'domain')
    # ==================== Transfinte curves ====================
    # Inlet
    n_in = kwargs.get('n_in') if kwargs.get('n_in') else 70
    prog_in = kwargs.get('prog_in', 0.99)
    gmsh.model.mesh.setTransfiniteCurve(tag = inlet, numNodes=n_in, coef=prog_in)

    # Outlet
    n_out = kwargs.get('n_out') if kwargs.get('n_out') else 70
    prog_out = kwargs.get('prog_out', 0.99)
    gmsh.model.mesh.setTransfiniteCurve(tag = outlet, numNodes=n_out, coef=prog_out)

    # Bed
    n_bed = kwargs.get('n_bed') if kwargs.get('n_bed') else 200
    prog_bed = kwargs.get('prog_bed', 1)
    gmsh.model.mesh.setTransfiniteCurve(tag = bed, numNodes=n_bed, coef=prog_bed)

    # Free surface
    n_fs = kwargs.get('n_fs') if kwargs.get('n_fs') else 200
    prog_fs = kwargs.get('prog_fs', 1)
    gmsh.model.mesh.setTransfiniteCurve(tag = fs, numNodes=n_fs, coef=prog_fs)

    # Airfoil
    prog_airfoil = kwargs.get('prog_airfoil', 1)
    for line in lines:
        gmsh.model.mesh.setTransfiniteCurve(tag = line, numNodes=n_airfoil//len(lines), coef=prog_airfoil)

    # ==================== Meshing and writing model ====================
    gmsh.model.mesh.generate(2)

    if kwargs.get('test', False):
        gmsh.fltk.run()

    if kwargs.get("write", False):
        out_name = kwargs.get('o', "naca")
        file_type = kwargs.get("file_type", "msh")
        gmsh.write(out_name + "." + file_type)

    # ==================== Converting to meshio ====================
    gmsh.write("temp.msh")

    mesh = meshio.read("temp.msh")
    os.system("rm temp.msh")
    
    gmsh.finalize()

    return mesh


def meshio_to_fd(mesh: meshio.Mesh):
    """
    Converts a meshio mesh to a firedrake mesh
    """

    # Write a temperary gmsh file from meshio mesh
    meshio.write("temp.msh", mesh, file_format="gmsh22")

    # Read that gmsh file as a firedrake mesh
    fd_mesh = fd.Mesh("temp.msh")

    # Delete the temperary gmsh file
    os.system("rm temp.msh")

    # Return the firedrake mesh
    return fd_mesh




if __name__ == "__main__":
    mesh = naca_mesh("0012", write = True, o = "test", n_in = 70, n_out = 70, n_bed = 70, n_fs = 300)

    plot_mesh(mesh)