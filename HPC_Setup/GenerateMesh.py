import numpy as np
import os


P = 3
meshSettings = {
    "airfoilNumber": "0012",
    "alpha_deg": 5,
    "circle": True,

    "xlim": (-8.5,19.5),
    "y_bed": -4,

    "scale": 1,

    "h": 1.034,
    "interface_ratio": 10,
    "nAirfoil": "calculated down below for a ration that Morten found, this seemed stable, but fast",
    "centerOfAirfoil": (0.5,0.0),

    "nFS": int(300),
    "nUpperSides": "Calculated down below to make upper elemets square (if they were not triangular xD)",
    "nLowerInlet": "calculated down below for a ration that Morten found, this seemed stable, but fast",
    "nLowerOutlet": "calculated down below for a ration that Morten found, this seemed stable, but fast",
    "nBed": "calculated down below for a ration that Morten found, this seemed stable, but fast",
    "test": True
    }

meshSettings["nLowerInlet"] = int( meshSettings["nFS"]/10 )
meshSettings["nLowerOutlet"] = int( meshSettings["nFS"]/10 )
meshSettings["nAirfoil"] = int( meshSettings["nFS"]/2 )
meshSettings["nBed"] = int( meshSettings["nFS"]/2 )
def calculateNUpperSides(meshSettings):
    nFS = meshSettings["nFS"]
    xlim = meshSettings["xlim"]
    h = meshSettings["h"]
    meshSettings["nUpperSides"] =  int( nFS/(xlim[1]-xlim[0]))
    return None
calculateNUpperSides(meshSettings)

def getMeshSettings():
    return meshSettings
    

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


def createFSMesh(airfoil: str, alpha: float, meshSettings: dict) -> list[np.float64, tuple]:
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
    import gmsh
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

    gmsh.write("mesh.msh")

    gmsh.finalize()

    ylim = (y_bed, h + y_interface)
    y_data = np.array([y_interface, *ylim])
    np.save("y_data", y_data)

    del(gmsh)
    print("Mesh Saved")


if __name__ =="__main__":
    createFSMesh("0012", meshSettings["alpha_deg"], meshSettings)
    import firedrake as fd
    mesh = fd.Mesh("mesh.msh")

    V = fd.FunctionSpace(mesh, "CG", P)
    print("DOF: ", V.dof_count)


    

        
