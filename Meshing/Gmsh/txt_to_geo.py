import gmsh
import sys
import os
import numpy as np
import argparse


# ==================== Handling the input arguments ====================
# Create the parser
parser = argparse.ArgumentParser(description='Converts a txt file with NACA coordinates to a Gmsh geo file')
parser.add_argument('filename', type=str, help='Name of the txt file with NACA coordinates')
parser.add_argument('-write', action='store_true', help='Write the geo file')
parser.add_argument('-nopopup', action='store_true', help='Do not launch the GUI')
parser.add_argument('-o', type=str, help='Name of the output geo file, default: <filename>.geo')
parser.add_argument('-n', type=int, help='Number of points in the airfoil, default: 200')
parser.add_argument('-alpha', type=float, help='Angle of attack in degrees')
parser.add_argument('-scale', type=float, help='Element scale factor')
args = parser.parse_args()

# ==================== Creating the geometry ====================
# Initialize the model
gmsh.initialize()

# Simple rectangular domain
xmin = -5
xmax = 10
ymin = -5
ymax = 1
scale = 1 if args.scale is None else args.scale
gmsh.model.geo.addPoint(xmin, ymin, 0, scale, tag = 1)
gmsh.model.geo.addPoint(xmin, ymax, 0, scale, tag = 2)
gmsh.model.geo.addPoint(xmax, ymin, 0, scale, tag = 3)
gmsh.model.geo.addPoint(xmax, ymax, 0, scale, tag = 4)

# Inlet
inlet = gmsh.model.geo.addLine(1, 2, tag = 1)
# Outlet
outlet = gmsh.model.geo.addLine(3, 4, tag = 2)
# Walls
top = gmsh.model.geo.addLine(2, 4, tag = 3)
bottom = gmsh.model.geo.addLine(1, 3, tag = 4)

# Line loop
boundary_loop = gmsh.model.geo.addCurveLoop([inlet, top, -outlet, -bottom], tag = 1)

# ==================== Handling the airfoil ====================
# Check if the file exists
filename = args.filename
if filename is None:
    print("Please provide the name of the txt file with NACA coordinates")
    sys.exit(1)
elif not os.path.exists(filename):
    print("The file does not exist")
    sys.exit(1)
elif filename[-4:] != ".txt":
    print("The file is not a txt file")
    sys.exit(1)

# Load the coordinates
coords = np.loadtxt(args.filename)

# Taking care of AOA
alpha = 0 if args.alpha is None else args.alpha
rotation_matrix = np.array([[np.cos(np.radians(alpha)), -np.sin(np.radians(alpha))],
                            [np.sin(np.radians(alpha)), np.cos(np.radians(alpha))]])
coords = np.dot(coords, rotation_matrix)

# Add 3. column with zeros
if coords.shape[1] == 2:
    coords = np.hstack((coords, np.zeros((coords.shape[0], 1))))

# Create points
points = []
n = 200 if args.n is None else args.n
for idx, coord in enumerate(coords[::200//n]):
    point = gmsh.model.geo.addPoint(coord[0], coord[1], coord[2], scale, tag = 4+idx+1)
    points.append(point)

# Create lines
lines = []
for idx in range(len(points)-1):
    line = gmsh.model.geo.addLine(points[idx], points[idx+1], tag = 4+idx + 1)
    lines.append(line)

line = gmsh.model.geo.addLine(points[-1], points[0], tag = 4+len(points))
lines.append(line)

airfoil_loop = gmsh.model.geo.addCurveLoop(lines, tag = 2)

# Create geometric surface
surface = gmsh.model.geo.addPlaneSurface([boundary_loop, airfoil_loop], tag = 1)

# ===================== ADDING PHYSICAL GROUPS =====================
# Add physical groups
gmsh.model.geo.addPhysicalGroup(1, [inlet], tag=101)  # Inlet
gmsh.model.setPhysicalName(1, 101, "Inlet")

gmsh.model.geo.addPhysicalGroup(1, [outlet], tag=102)  # Outlet
gmsh.model.setPhysicalName(1, 102, "Outlet")

gmsh.model.geo.addPhysicalGroup(1, [top], tag=103)  # Top wall
gmsh.model.setPhysicalName(1, 103, "Top")

gmsh.model.geo.addPhysicalGroup(1, [bottom], tag=104)  # Bottom wall
gmsh.model.setPhysicalName(1, 104, "Bottom")

# Airfoil boundary (all airfoil lines together)
gmsh.model.geo.addPhysicalGroup(1, lines, tag=105)  # Airfoil
gmsh.model.setPhysicalName(1, 105, "Airfoil")

# Physical Surface (Computational Domain)
gmsh.model.geo.addPhysicalGroup(2, [surface], tag=201)  # Plane Surface
gmsh.model.setPhysicalName(2, 201, "Domain")


# Synchronize the model
gmsh.model.geo.synchronize()

if args.write:
    out_name = args.o if args.o is not None else filename[:-4]
    gmsh.write(out_name + ".geo_unrolled")
    os.rename(out_name + ".geo_unrolled", out_name + ".geo")


# Launch the GUI to see the results:
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()