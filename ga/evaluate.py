from geometry.raybundle import RayBundle
from geometry.ts import GeometryParams
from foam.runner import run_case
from foam.extractCD import extract_latest_cd

def evaluate(params: GeometryParams, density=10) -> float:
    bundle = RayBundle(width=1, height=1, depth=1, density=density)