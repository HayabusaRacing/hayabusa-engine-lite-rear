from geometry.rayBundle import RayBundle
from geometry.geometryParams import GeometryParams
from foam.setupCase import setup_case
from foam.runner import OpenFOAMParallelRunner
from foam.extractCD import extract_latest_cd
from config import (BASE_DIR, CASE_DIR, TMP_DIR, MESH_WIDTH, MESH_HEIGHT, MESH_DEPTH, 
                   MESH_DENSITY, MESH_CENTER, MESH_UNIT, MIN_WING_DIMENSIONS,
                   DIMENSION_PENALTY_WEIGHT, VOLUME_REWARD_WEIGHT, SMOOTHNESS_REWARD_WEIGHT, 
                   CD_WEIGHT, EXPECTED_VOLUME_RANGE)

import trimesh
import numpy as np
from pathlib import Path


def calculate_mesh_smoothness(mesh):
    """
    Calculate smoothness penalty that only penalizes spiky geometries.
    Returns 0 for smooth wings, increasing penalty for spiky features.
    """
    try:
        face_normals = mesh.face_normals
        face_adjacency = mesh.face_adjacency
        
        if len(face_adjacency) == 0:
            return 0.0
        
        angles = []
        for adj in face_adjacency:
            face1_normal = face_normals[adj[0]]
            face2_normal = face_normals[adj[1]]
            dot_product = np.clip(np.dot(face1_normal, face2_normal), -1.0, 1.0)
            angle = np.arccos(dot_product)
            angles.append(angle)
        
        angles = np.array(angles)
        
        # Smoothness threshold: angles below this are considered "smooth enough"
        smooth_threshold = np.pi / 6  # 30 degrees
        
        # Only penalize angles above threshold
        spiky_angles = angles[angles > smooth_threshold]
        
        if len(spiky_angles) == 0:
            return 0.0  # No spiky features, no penalty
        
        # Penalty based on: 1) number of spiky edges, 2) severity of spikes
        spike_severity = np.mean(spiky_angles - smooth_threshold)
        spike_fraction = len(spiky_angles) / len(angles)
        
        # Combined penalty: severity * prevalence
        smoothness_penalty = spike_severity * spike_fraction
        
        return smoothness_penalty
        
    except Exception as e:
        print(f"Smoothness calculation error: {e}")
        return 1.0


def check_dimension_constraints(mesh, min_dims):
    bounds = mesh.bounds
    dimensions = bounds[1] - bounds[0]
    
    penalties = 0.0
    if dimensions[0] < min_dims['x']:
        penalties += (min_dims['x'] - dimensions[0]) / min_dims['x']
    
    if dimensions[1] < min_dims['y']:
        penalties += (min_dims['y'] - dimensions[1]) / min_dims['y']
        
    if dimensions[2] < min_dims['z']:
        penalties += (min_dims['z'] - dimensions[2]) / min_dims['z']
    
    return penalties


def calculate_bounding_box_volume(mesh):
    bounds = mesh.bounds
    dimensions = bounds[1] - bounds[0]
    volume = dimensions[0] * dimensions[1] * dimensions[2]
    return volume


def calculate_fitness(cd, mesh):
    drag_component = CD_WEIGHT * cd
    try:
        volume = mesh.volume
        if volume <= 0 or volume < 1e-12 or not mesh.is_watertight:
            volume = calculate_bounding_box_volume(mesh)
            volume_source = "bounding_box"
        else:
            volume_source = "mesh_volume"
    except:
        volume = calculate_bounding_box_volume(mesh)
        volume_source = "bounding_box"
    volume_normalized = (volume - EXPECTED_VOLUME_RANGE[0]) / (EXPECTED_VOLUME_RANGE[1] - EXPECTED_VOLUME_RANGE[0])
    volume_normalized = max(0, min(1, volume_normalized))
    volume_component = VOLUME_REWARD_WEIGHT * volume_normalized
    smoothness = calculate_mesh_smoothness(mesh)
    smoothness_component = SMOOTHNESS_REWARD_WEIGHT * smoothness
    dimension_penalty = check_dimension_constraints(mesh, MIN_WING_DIMENSIONS)
    dimension_component = DIMENSION_PENALTY_WEIGHT * dimension_penalty
    fitness = drag_component + volume_component + smoothness_component + dimension_component
    fitness_breakdown = {
        "Cd": cd,
        "volume": volume,
        "volume_source": volume_source,
        "smoothness": smoothness,
        "dimension_penalty": dimension_penalty,
        "drag_component": drag_component,
        "volume_component": volume_component,
        "smoothness_component": smoothness_component,
        "dimension_component": dimension_component,
        "total_fitness": fitness
    }
    
    return fitness, fitness_breakdown

def evaluate(params: GeometryParams) -> float:
    try:
        setup_case(base_dir=BASE_DIR, case_dir=CASE_DIR)
        
        bundle = RayBundle(width=MESH_WIDTH, height=MESH_HEIGHT, depth=MESH_DEPTH, density=MESH_DENSITY, center=MESH_CENTER, unit=MESH_UNIT)
        bundle.set_ts(params.ts)
        bundle.export_stl(TMP_DIR / "wing.stl")
        mesh_wing = trimesh.load(TMP_DIR / "wing.stl")
        mesh_body = trimesh.load(BASE_DIR / "constant" / "triSurface" / "mainBodyNoWing.stl")
        combined = trimesh.util.concatenate([mesh_body, mesh_wing])
        combined.export(CASE_DIR / "constant" / "triSurface" / "mainBody.stl")
        runner = OpenFOAMParallelRunner(case_dir=CASE_DIR, n_proc=6)
        runner.run_all()
        result = extract_latest_cd()
        cd = result["Cd"]
        fitness, fitness_breakdown = calculate_fitness(cd, mesh_wing)
        params.fitness = fitness
        params.fitness_breakdown = fitness_breakdown
        
        return fitness
        
    except Exception as e:
        print(f"Evaluation error: {e}")
        return float('inf')