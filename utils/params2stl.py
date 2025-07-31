import argparse
import json
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from geometry.airfoilLayers import airfoilLayers
from config import (AIRFOIL_DENSITY, AIRFOIL_WING_SPAN, AIRFOIL_WING_CHORD, AIRFOIL_FILES,
                   AIRFOIL_Y_CENTER, AIRFOIL_X_CENTER, AIRFOIL_Z_CENTER,
                   AIRFOIL_SURFACE_DEGREE_U, AIRFOIL_SURFACE_DEGREE_V, AIRFOIL_SAMPLE_RESOLUTION)

parser = argparse.ArgumentParser()
parser.add_argument("--gen", type=int, default=0)
parser.add_argument("--child", type=int, default=0)

args = parser.parse_args()

gen_str = "generation" + "{:03}".format(args.gen)
child_str = "child" + "{:03}".format(args.child)

output_file_name = f"wing_{gen_str}_{child_str}.stl"

path_str = "../results/" + gen_str + "/" + child_str + "/ts.json"

with open(path_str, "r") as f:
    data = json.load(f)

wing_generator = airfoilLayers(
    density=AIRFOIL_DENSITY, 
    wing_span=AIRFOIL_WING_SPAN, 
    wing_chord=AIRFOIL_WING_CHORD,
    y_center=AIRFOIL_Y_CENTER,
    x_center=AIRFOIL_X_CENTER,
    z_center=AIRFOIL_Z_CENTER,
    surface_degree_u=AIRFOIL_SURFACE_DEGREE_U,
    surface_degree_v=AIRFOIL_SURFACE_DEGREE_V,
    sample_resolution=AIRFOIL_SAMPLE_RESOLUTION
)
wing_generator.create_geometry_from_array(data, AIRFOIL_FILES, output_file_name)