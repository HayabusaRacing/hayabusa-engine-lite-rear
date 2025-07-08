import argparse
import json
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from geometry.rayBundle import RayBundle

parser = argparse.ArgumentParser()
parser.add_argument("--gen", type=int, default=0)
parser.add_argument("--child", type=int, default=0)

args = parser.parse_args()

gen_str = "generation" + "{:03}".format(args.gen)
child_str = "child" + "{:03}".format(args.child)

path_str = "../results/" + gen_str + "/" + child_str + "/ts.json"

with open(path_str, "r") as f:
    data = json.load(f)

bundle = RayBundle(width=0.07, height=0.0255, depth=0.02, density=10, center=[0, -0.0625, 0.015], unit='m')
bundle.set_ts(data)
bundle.export_stl("wing.stl")