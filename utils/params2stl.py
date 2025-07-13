import argparse
import json
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from geometry.rayBundle import RayBundle
from config import MESH_WIDTH, MESH_HEIGHT, MESH_DEPTH, MESH_DENSITY, MESH_CENTER, MESH_UNIT

parser = argparse.ArgumentParser()
parser.add_argument("--gen", type=int, default=0)
parser.add_argument("--child", type=int, default=0)

args = parser.parse_args()

gen_str = "generation" + "{:03}".format(args.gen)
child_str = "child" + "{:03}".format(args.child)

path_str = "../results/" + gen_str + "/" + child_str + "/ts.json"

with open(path_str, "r") as f:
    data = json.load(f)

bundle = RayBundle(width=MESH_WIDTH, height=MESH_HEIGHT, depth=MESH_DEPTH, density=MESH_DENSITY, center=MESH_CENTER, unit=MESH_UNIT)
bundle.set_ts(data)
bundle.export_stl("wing.stl")