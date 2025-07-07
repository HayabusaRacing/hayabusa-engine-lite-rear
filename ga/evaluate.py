from geometry.raybundle import RayBundle
from geometry.ts import GeometryParams
from foam.setupCase import setup_case
from foam.runner import run_case
from foam.extractCD import extract_latest_cd

def evaluate(params: GeometryParams, density=10) -> float:
    setup_case(base_dir="../baseCase", case_dir="../case")
    
    bundle = RayBundle(width=0.07, height=0.0255, depth=0.02, density=density, center=[0, -0.0625, 0.015], unit='m')
    bundle.set_ts(params.ts)
    bundle.export_stl("../tmp/wing.stl")
    mesh_wing = trimesh.load("../tmp/wing.stl")
    mesh_body = trimesh.load("../baseCase/constant/triSurface/mainBodyNoWing.stl")
    combined = trimesh.util.concatenate([mesh_body, mesh_wing])
    combined.export("../case/constant/triSurface/mainBody.stl")
    
    runner = OpenFOAMParallelRunner(case_dir="../case", n_proc=6)
    runner.run_all()
    
    result = extract_latest_cd()
    cd = result["Cd"]
    
    params.fitness = cd
    
    return cd