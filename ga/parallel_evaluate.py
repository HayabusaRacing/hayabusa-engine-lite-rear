from geometry.airfoilLayers import airfoilLayers
from geometry.geometryParams import GeometryParams
from foam.setupCase import setup_case
from foam.runner import OpenFOAMParallelRunner
from foam.extractCD import extract_latest_cd
from ga.evaluate import calculate_fitness
from utils.logging_utils import setup_logging
from config import (BASE_DIR, CASE_DIR, TMP_DIR, PARALLEL_EVALUATIONS, CORES_PER_CFD,
                   AIRFOIL_DENSITY, AIRFOIL_WING_SPAN, AIRFOIL_WING_CHORD, AIRFOIL_FILES,
                   AIRFOIL_Y_CENTER, AIRFOIL_X_CENTER, AIRFOIL_Z_CENTER,
                   AIRFOIL_SURFACE_DEGREE_U, AIRFOIL_SURFACE_DEGREE_V, AIRFOIL_SAMPLE_RESOLUTION,
                   AIRFOIL_CENTER_FIXED)

import trimesh
import concurrent.futures
import multiprocessing
from pathlib import Path
import shutil
import os


def setup_individual_case(base_dir, case_id):
    case_dir = CASE_DIR / f"indiv_{case_id}"
    if os.path.exists(case_dir):
        shutil.rmtree(case_dir)
    shutil.copytree(base_dir, case_dir)
    
    # Auto-update decomposeParDict
    from foam.setupCase import update_decompose_par_dict
    update_decompose_par_dict(case_dir, CORES_PER_CFD)
    
    return case_dir


def evaluate_single(params_and_id):
    params, case_id = params_and_id
    case_dir = None
    tmp_dir = None
    
    # Setup logging for this case
    logger, log_file = setup_logging(case_id)
    
    try:
        logger.info(f"Starting evaluation for case {case_id}")
        
        case_dir = setup_individual_case(BASE_DIR, case_id)
        tmp_dir = TMP_DIR / f"indiv_{case_id}"
        tmp_dir.mkdir(exist_ok=True)
        
        logger.info(f"Case directory created: {case_dir}")
        
        # Use airfoilLayers method
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
        wing_stl_path = tmp_dir / "wing.stl"
        wing_generator.create_geometry_from_array(
            params.ts, AIRFOIL_FILES, str(wing_stl_path)
        )
        logger.info(f"Wing STL exported using airfoilLayers: {wing_stl_path}")
        
        mesh_wing = trimesh.load(wing_stl_path)
        mesh_body = trimesh.load(BASE_DIR / "constant" / "triSurface" / "mainBodyNoWing.stl")
        combined = trimesh.util.concatenate([mesh_body, mesh_wing])
        combined.export(case_dir / "constant" / "triSurface" / "mainBody.stl")
        logger.info("Combined mesh created")
        
        runner = OpenFOAMParallelRunner(case_dir=case_dir, n_proc=CORES_PER_CFD, case_id=case_id)
        logger.info("Starting OpenFOAM simulation")
        success = runner.run_all()
        
        if not success:
            logger.error(f"OpenFOAM failed for case {case_id}")
            return float('inf'), None
        
        logger.info("OpenFOAM simulation completed successfully")
        
        cd_file = case_dir / "postProcessing" / "forceCoeffs1" / "0" / "coefficient.dat"
        if not cd_file.exists():
            logger.error(f"coefficient.dat not found for case {case_id}")
            return float('inf'), None
            
        result = extract_latest_cd(str(cd_file.parent))
        cd = result["Cd"]
        logger.info(f"Extracted Cd: {cd}")
        
        # Validate Cd value - reject unrealistic results
        if cd < 0.0001 or cd > 0.1 or abs(cd) > 1e10:
            logger.error(f"Unrealistic Cd value detected: {cd}. Expected range: 0.0001-0.1")
            return float('inf'), None
        
        fitness, fitness_breakdown = calculate_fitness(cd, mesh_wing)
        logger.info(f"Fitness calculated: {fitness}")
        
        return fitness, fitness_breakdown
        
    except Exception as e:
        logger.error(f"Evaluation error for case {case_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return float('inf'), None
        
    finally:
        # Cleanup in finally block to ensure it always happens
        try:
            if case_dir and case_dir.exists():
                shutil.rmtree(case_dir)
                logger.info(f"Cleaned up case directory: {case_dir}")
            if tmp_dir and tmp_dir.exists():
                shutil.rmtree(tmp_dir)
                logger.info(f"Cleaned up tmp directory: {tmp_dir}")
        except Exception as cleanup_error:
            logger.error(f"Cleanup error for case {case_id}: {cleanup_error}")


def evaluate_batch_parallel(params_list):
    with concurrent.futures.ProcessPoolExecutor(max_workers=PARALLEL_EVALUATIONS) as executor:
        params_with_ids = [(params, i) for i, params in enumerate(params_list)]
        results = list(executor.map(evaluate_single, params_with_ids))
    
    for i, (fitness, fitness_breakdown) in enumerate(results):
        if fitness != float('inf') and fitness_breakdown is not None:
            params_list[i].fitness = fitness
            params_list[i].fitness_breakdown = fitness_breakdown
        else:
            params_list[i].fitness = fitness
            params_list[i].fitness_breakdown = {"fitness": fitness}
    
    return [result[0] for result in results]
