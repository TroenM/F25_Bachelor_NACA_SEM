import gmsh
import numpy as np


def naca_gmsh(airfoil: str, alpha: float = 0, xlim: tuple = (-7,13), ylim: tuple = (-2,1), **kwargs):
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
    gmsh.initialize()

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
    coords = np.loadtxt(airfoil)
    poa = kwargs.get('poa', 200)

    # AoA
    alpha = np.deg2rad(alpha)
    rot_matrix = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    coords = np.dot(coords, rot_matrix)

    points = []
    for idx, coord in enumerate(coords[::len(coords)//poa]):
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
    gmsh.model.addPhysicalGroup(1, [airfoil_line], tag=5)
    gmsh.model.setPhysicalName(1, 5, 'airfoil')

    # Domain
    gmsh.model.addPhysicalGroup(2, [1], tag=6)
    gmsh.model.setPhysicalName(2, 6, 'domain')
    # ==================== Transfinte curves ====================
    # Inlet
    n_in = kwargs.get('n_in', 10*7)
    prog_in = kwargs.get('prog_in', 0.99)
    gmsh.model.mesh.setTransfiniteCurve(tag = 1, numNodes=n_in, coef=prog_in)

    # Outlet
    n_out = kwargs.get('n_out', 10*7)
    prog_out = kwargs.get('prog_out', 0.99)
    gmsh.model.mesh.setTransfiniteCurve(tag = 2, numNodes=n_out, coef=prog_out)

    # Bed
    n_bed = kwargs.get('n_bed', 40*8)
    prog_bed = kwargs.get('prog_bed', 1)
    gmsh.model.mesh.setTransfiniteCurve(tag = 3, numNodes=n_bed, coef=prog_bed)

    # Free surface
    n_fs = kwargs.get('n_fs', 100*5)
    prog_fs = kwargs.get('prog_fs', 1)
    gmsh.model.mesh.setTransfiniteCurve(tag = 4, numNodes=n_fs, coef=prog_fs)

    # Airfoil
    n_airfoil = kwargs.get('n_airfoil', 50*6)
    for line in lines:
        gmsh.model.mesh.setTransfiniteCurve(tag = line, numNodes=n_airfoil//len(lines), coef=1)

    # ==================== Finalize the model ====================
    gmsh.model.mesh.generate(2)

    if kwargs.get('test', False):
        gmsh.fltk.run()

    gmsh.finalize()

    return None


if __name__ == '__main__':
    naca_gmsh('NACA_0015.txt', test = True, kwargs = {"poa" : 120, "n_airfoil" : 120, "n_in" : 70, "n_out" : 70, "n_bed" : 320, "n_fs" : 500})





