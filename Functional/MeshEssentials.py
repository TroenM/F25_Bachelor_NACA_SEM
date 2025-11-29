import firedrake as fd
import numpy as np
import gmsh
import os

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

    # If an equal amount of nodes is used to approximate the airfoil
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

    # If an odd amount of nodes is used to approximate the airfoil
    elif (n//2)*2 != n:

        # Create a linspace with points spaced equally between 0 and pi for the lower surface of the airfoil with one more point than for the upper part
        beta = np.linspace(0,np.pi,(n//2)+1)

        # Convert this linspace to an array with points that are the most concentrated at the edges of the airfoil.
        x = (1-np.cos(beta))/2
        
        # Define the thickness of the airfoil distrebution in terms of x
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)

        # Define the camber in terms of x
        yc = np.where(x < p, m/p**2 * (2*p*x - x**2), m/(1-p)**2 * ((1-2*p) + 2*p*x - x**2))

        # Define the lower part of the airfoil
        lower = np.hstack((x.reshape(-1,1), (yc - yt).reshape(-1,1)))[::-1][:-1]

        # Create a linspace with points spaced equally between 0 and pi for the upper surface of the airfoil
        beta = np.linspace(0,np.pi,(n//2)+2)

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

def naca_mesh(airfoil: str, alpha: float, meshSettings) -> list[fd.MeshGeometry, np.float64, tuple]:
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
    
    ** meshSettings:
        - n_in, n_out, n_bed, n_fs, n_airfoil: int
            Number of points in the inlet, outlet, bed, front step and airfoil
        - prog_in, prog_out, prog_bed, prog_fs: float
            Progression in the inlet, outlet, bed and front step
        - scale: float
            Element scale factor
        
    """

    # ==================== Initializing the model ====================

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)

    # Gathering xlim and y_bed from mesh settings
    xmin, xmax = meshSettings["xlim"]
    y_bed = meshSettings["y_bed"]

    # The insulating layer will be 1/10 of the distance from the airfoil to the surface
    h = meshSettings['h']
    interface_ratio = meshSettings.get("interface_ratio", 10)
    naca_eps = h/interface_ratio
    h -= naca_eps


    n_airfoil = meshSettings["nAirfoil"]
    n_fs = meshSettings["nFS"]
    n_bed = meshSettings["nBed"]
    n_in = meshSettings["nLowerInlet"] + meshSettings["nUpperSides"]
    n_out = meshSettings["nLowerOutlet"] + meshSettings["nUpperSides"]


    # ==================== Handling the airfoil ====================
    coords = naca_4digit(airfoil, n = n_airfoil, alpha=alpha, position_of_center=meshSettings.get("center_of_airfoil",np.array([0.5,0])))
    y_interface = np.max(coords[:,1]) + naca_eps

    # Defining ylim
    ylim = (y_bed, h + y_interface)
    ymin, ymax = ylim

    scale = meshSettings.get('scale', 1.034)
    gmsh.model.geo.addPoint(xmin, ymin, 0, scale, tag=1) # bottom left
    gmsh.model.geo.addPoint(xmin, ymax, 0, scale, tag=2) # top left
    gmsh.model.geo.addPoint(xmax, ymin, 0, scale, tag=3) # bottom right
    gmsh.model.geo.addPoint(xmax, ymax, 0, scale, tag=4) # top right

    inlet = gmsh.model.geo.addLine(1, 2, tag=1)
    outlet = gmsh.model.geo.addLine(3, 4, tag=2)
    bed = gmsh.model.geo.addLine(1, 3, tag=3)
    fs = gmsh.model.geo.addLine(2, 4, tag=4)

    boundary_loop = gmsh.model.geo.addCurveLoop([inlet, fs, -outlet, -bed], tag=1)

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
    prog_in = meshSettings.get('prog_in', 0.99)
    gmsh.model.mesh.setTransfiniteCurve(tag = inlet, numNodes=n_in, coef=prog_in)

    # Outlet
    prog_out = meshSettings.get('prog_out', 0.99)
    gmsh.model.mesh.setTransfiniteCurve(tag = outlet, numNodes=n_out, coef=prog_out)

    # Bed
    prog_bed = meshSettings.get('prog_bed', 1)
    gmsh.model.mesh.setTransfiniteCurve(tag = bed, numNodes=n_bed, coef=prog_bed)

    # Free surface
    prog_fs = meshSettings.get('prog_fs', 1)
    gmsh.model.mesh.setTransfiniteCurve(tag = fs, numNodes=n_fs, coef=prog_fs)

    # Airfoil
    prog_airfoil = meshSettings.get('prog_airfoil', 1)
    for line in lines:
        gmsh.model.mesh.setTransfiniteCurve(tag = line, numNodes=n_airfoil//len(lines), coef=prog_airfoil)

    # ==================== Meshing and writing model ====================
    gmsh.model.mesh.generate(2)

    # if meshSettings.get('test', False):
    #     gmsh.fltk.run()

    # if meshSettings.get("write", False):
    #     out_name = meshSettings.get('o', "naca")
    #     file_type = meshSettings.get("file_type", "msh")
    #     gmsh.write(out_name + "." + file_type)

    # ==================== Converting to meshio ====================
    gmsh.write("temp.msh")

    mesh = fd.Mesh("temp.msh")
    os.system("rm temp.msh")
    
    gmsh.finalize()
    return mesh, y_interface, ylim

def shiftSurface(mesh : fd.Mesh, func_b : fd.Function, func_a : fd.Function) -> fd.MeshGeometry:
    '''
    params
    ---
    etaCurrent: callable 
        - eta of the current mesh
    etaNext: callable
        - eta of the outputtet mesh
    '''
    # Define p=1 functionspace
    V1 = fd.FunctionSpace(mesh, "CG", 1)

    coords = mesh.coordinates.dat.data
    
    # Find airfoil indecies
    naca_idx = V1.boundary_nodes(5)

    # Define maximal y value of coordinates on airfoil 
    M = np.max(coords[naca_idx][:,1])

    # Mask all points above the largest value of the airfoil
    point_mask = np.where((coords[:,1] > M))

    # Manipulate functions
    func_before = lambda x: func_b(x)-M
    func_after = lambda x: func_a(x)-M
    func_before = np.vectorize(func_before)
    func_after = np.vectorize(func_after)
    
    # Manipulate point values as stated in the report under methodology for free surface.
    coords[point_mask,1] -= M
    coords[point_mask,1] *= (func_after(coords[point_mask,0]) /  func_before(coords[point_mask,0]))
    coords[point_mask,1] += M

    # Set new coordinates of mesh
    mesh.coordinates.dat.data[:] = coords

    # Return mesh
    return mesh

def createFSMesh(airfoil: str, alpha: float, meshSettings: dict) -> list[fd.MeshGeometry, np.float64, tuple]:
    """
    Generates a mesh with a insulating ordered mesh above airfoil for better free surface computations

    meshSettings:
    ---
    xlim: (xmin, xmax)
    y_bed: ymin
    
    scale: float
        - Target mesh size at generated points
    
    h: float
        - Distance from surface to airfoil
    interface_ratio: float
        - The ratio h/(distance from naca_airfoil to interface line)
        - Default: 10
    
    nAirfoil: int
    centerOfAirfoil: (xc, yc)

    nFS: int
        - Number of nodes on free surface and interface line
    nBed: int
    nUpperSides: int
        - Number of nodes on insulating layer inlet and outlet
    nLowerOutlet: int
    nLowerInlet: int
    
    Returns:
    mesh: fd.mesh.MeshGeometry
    y_interface: np.float64
    ylim: (ymin, ymax)
        
    """

    # ===================== Variable Preparation =======================

    xmin, xmax = meshSettings["xlim"]
    y_bed = meshSettings["y_bed"]

    scale = meshSettings["scale"]

    # The insulating layer will be 9/10 of the distance from surface to airfoil
    h = meshSettings['h']
    interface_ratio = meshSettings.get("interface_ratio", 10)
    naca_eps = h/interface_ratio
    h -= naca_eps

    xc, yc = meshSettings["centerOfAirfoil"]
    n_airfoil = meshSettings["nAirfoil"]

    n_fs = meshSettings["nFS"]
    n_bed = meshSettings["nBed"]
    n_lower_inlet = meshSettings["nLowerInlet"]
    n_lower_outlet = meshSettings["nLowerOutlet"]
    n_upper_sides = meshSettings["nUpperSides"]
    

    naca_coords = naca_4digit(airfoil, n = n_airfoil, alpha=alpha, position_of_center=(xc, yc))
    y_interface = np.max(naca_coords[:,1]) + naca_eps

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    # ====================== Lower domain ======================
    # point Lower Top Left
    pLTL = gmsh.model.geo.addPoint(xmin, y_interface, 0, scale)
    pLBL = gmsh.model.geo.addPoint(xmin, y_bed, 0, scale)
    pLBR = gmsh.model.geo.addPoint(xmax, y_bed, 0, scale)
    pLTR = gmsh.model.geo.addPoint(xmax, y_interface, 0, scale)

    # Lines going counter clock wise
    lLInlet = gmsh.model.geo.addLine(pLTL, pLBL) # line Lower Inlet
    lLOutlet = gmsh.model.geo.addLine(pLBR, pLTR) # line Lower outlet
    lbed = gmsh.model.geo.addLine(pLBL, pLBR) # line bed
    lInterface = gmsh.model.geo.addLine(pLTR, pLTL) # line Interface

    lowerLoop = gmsh.model.geo.addCurveLoop([lLInlet, lbed, lLOutlet, lInterface])

    # ===================== Upper domain ========================
    pUTL = gmsh.model.geo.addPoint(xmin, h + y_interface, 0, scale)
    pUTR = gmsh.model.geo.addPoint(xmax, h + y_interface, 0, scale)

    lUInlet = gmsh.model.geo.addLine(pLTL, pUTL)
    lFreeSurfaces = gmsh.model.geo.addLine(pUTL, pUTR)
    lUOutlet = gmsh.model.geo.addLine(pUTR, pLTR)

    upperLoop = gmsh.model.geo.addCurveLoop([lUInlet, lFreeSurfaces, lUOutlet, lInterface])
    # ===================== Airfoil =========================
    naca_points = []
    for coord in naca_coords:
        naca_points.append(gmsh.model.geo.addPoint(coord[0], coord[1], 0, scale))
    
    naca_lines = []
    for idx in range(len(naca_points)-1):
        line = gmsh.model.geo.addLine(naca_points[idx], naca_points[idx+1])
        naca_lines.append(line)
    
    last_naca_line = gmsh.model.geo.addLine(naca_points[-1], naca_points[0])
    naca_lines.append(last_naca_line)

    airfoil_line = gmsh.model.geo.addCurveLoop(naca_lines, tag=5)

    # ================== Generating Surface ==================
    # Create the surface
    lowerDomain = gmsh.model.geo.addPlaneSurface([lowerLoop, airfoil_line])
    upperDomain = gmsh.model.geo.addPlaneSurface([upperLoop])
    gmsh.model.geo.synchronize()

    # ================== Transfine Constraints ============

    # Ensure fs and interface shares number of nodes.
    gmsh.model.mesh.setTransfiniteCurve(lFreeSurfaces, n_fs, coef=1)
    gmsh.model.mesh.setTransfiniteCurve(tag=lInterface, numNodes=n_fs, coef=1)

    # Ensure upper inlet and outlet shares number of nodes
    gmsh.model.mesh.setTransfiniteCurve(lUInlet, n_upper_sides, coef=1)
    gmsh.model.mesh.setTransfiniteCurve(lUOutlet, n_upper_sides, coef=1)

    # Structure upper domain
    gmsh.model.mesh.setTransfiniteSurface(tag=upperDomain)

    # Set remaining line numbers
    gmsh.model.mesh.setTransfiniteCurve(lbed, n_bed, coef=1)
    gmsh.model.mesh.setTransfiniteCurve(lLInlet, n_lower_inlet, coef = 1)
    gmsh.model.mesh.setTransfiniteCurve(lLOutlet, n_lower_outlet, coef = 1)

    for line in naca_lines:
        gmsh.model.mesh.setTransfiniteCurve(tag = line, numNodes=n_airfoil//len(naca_lines), coef=1)

    # ================== Physical Elements ================
    gmsh.model.addPhysicalGroup(1, [lUInlet, lLInlet], tag = 1, name = "Inlet")
    gmsh.model.addPhysicalGroup(1, [lUOutlet, lLOutlet], tag = 2, name = "Outlet")
    gmsh.model.addPhysicalGroup(1, [lbed], tag = 3, name = "Bed")
    gmsh.model.addPhysicalGroup(1, [lFreeSurfaces], tag = 4, name = "FreeSurface")
    gmsh.model.addPhysicalGroup(1, naca_lines, tag = 5, name = "Airfoil")
    
    gmsh.model.addPhysicalGroup(2, [upperDomain, lowerDomain], tag = 1, name = "Fluid")
    # =================== Generate Mesh ==================
    # gmsh.model.mesh.setRecombine(2, upperDomain)

    gmsh.model.mesh.generate(2)

    if meshSettings.get("show", False):
        gmsh.fltk.run()

    gmsh.write("temp.msh")

    mesh = fd.Mesh("temp.msh")
    os.system("rm temp.msh")

    gmsh.finalize()
    ylim = (y_bed, h + y_interface)
    return mesh, y_interface, ylim

    






if __name__ == "__main__":

    meshSettings = {
    "xlim": (-8,27),
    "y_bed": -4,

    "scale": 1,
    
    "h": 1,
    "interface_ratio": 10,
    "nAirfoil": 100,
    "centerOfAirfoil": (0,0.5),

    "nFS": 300,
    "nUpperSides": 20,
    "nLowerInlet": 20,
    "nLowerOutlet": 20,
    "nBed": 100,

    "show": True
    }

    mesh, y_interface, ylim = createFSMesh("0012", 5, meshSettings=meshSettings)
    print(type(mesh))

    # Mesh settings
    airfoilNumber = "0012"
    xlim = (-7, 13)
    ylim = (-4, 2)
    nIn = 20
    nOut = 20
    nBed = 50
    nFS = 50 
    nAirfoil = 100