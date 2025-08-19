"""
Test script to verify bounding box penalty calculation for wing geometries.
This script will:
1. Generate multiple wing shapes in different positions
2. Check if they're inside or outside the bounding box
3. Export STL files for visual inspection
4. Create a visualization of the bounding box itself
"""

import sys
import os
import numpy as np
import trimesh
from pathlib import Path
from geometry.airfoilLayers import airfoilLayers
from geometry.geometryParams import GeometryParams
from ga.evaluate import calculate_bounding_box_penalty
from config import (BOUNDING_BOX_LIMITS, AIRFOIL_FILES, TMP_DIR,
                   AIRFOIL_DENSITY, AIRFOIL_WING_SPAN, AIRFOIL_WING_CHORD,
                   AIRFOIL_X_CENTER, AIRFOIL_Y_CENTER, AIRFOIL_Z_CENTER,
                   AIRFOIL_SURFACE_DEGREE_U, AIRFOIL_SURFACE_DEGREE_V, AIRFOIL_SAMPLE_RESOLUTION)

# Create test directory if it doesn't exist
TEST_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "test_results"
TEST_DIR.mkdir(exist_ok=True)

def create_bounding_box_mesh():
    """Create a mesh representing the bounding box limits for visualization"""
    # Extract bounds from config
    x_min = BOUNDING_BOX_LIMITS['x_min']
    x_max = BOUNDING_BOX_LIMITS['x_max']
    y_min = BOUNDING_BOX_LIMITS['y_min']
    y_max = BOUNDING_BOX_LIMITS['y_max']
    z_min = BOUNDING_BOX_LIMITS['z_min']
    z_max = BOUNDING_BOX_LIMITS['z_max']
    
    # Create box vertices (8 corners)
    vertices = np.array([
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
        [x_min, y_max, z_max]
    ])
    
    # Define faces (6 faces with 2 triangles each)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom face
        [4, 5, 6], [4, 6, 7],  # Top face
        [0, 1, 5], [0, 5, 4],  # Front face
        [2, 3, 7], [2, 7, 6],  # Back face
        [0, 3, 7], [0, 7, 4],  # Left face
        [1, 2, 6], [1, 6, 5]   # Right face
    ])
    
    # Create mesh
    bbox_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Make it transparent (won't affect STL but useful for other formats)
    # if hasattr(bbox_mesh.visual, 'face_colors'):
    #     bbox_mesh.visual.face_colors = [100, 100, 100, 50]  # Semi-transparent gray
    
    # Export as STL
    bbox_mesh.export(TEST_DIR / "bounding_box.stl")
    print(f"Bounding box STL exported to {TEST_DIR / 'bounding_box.stl'}")
    
    return bbox_mesh

def create_test_wings():
    """Generate test wings with various positions and check if they're inside the box"""
    # Base wing generator
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
    
    # Create a basic params object
    base_params = GeometryParams()
    base_params.ts = [{'wing_type_idx': 0, 'pitch_angle': 0, 'y_offset': 0, 'z_offset': 0, 'scale': 1.0},
                      {'wing_type_idx': 1, 'pitch_angle': 0, 'y_offset': 0, 'z_offset': 0, 'scale': 1.0},
                      {'wing_type_idx': 0, 'pitch_angle': 0, 'y_offset': 0, 'z_offset': 0, 'scale': 1.0}]
    
    # Create a list of test cases with different offsets
    # We'll modify y and z offsets as these often cause bounding box issues
    test_cases = [
        {"name": "baseline", "y_offset": 0, "z_offset": 0},
        {"name": "y_min_violation", "y_offset": -0.05, "z_offset": 0},
        {"name": "y_max_violation", "y_offset": 0.05, "z_offset": 0},
        {"name": "z_min_violation", "y_offset": 0, "z_offset": -0.02},
        {"name": "z_max_violation", "y_offset": 0, "z_offset": 0.02},
        {"name": "corner_violation", "y_offset": 0.05, "z_offset": 0.02},
        {"name": "barely_inside", "y_offset": 0.001, "z_offset": 0.001}
    ]
    
    results = []
    
    for tc in test_cases:
        # Create a modified params object for this test case
        test_params = GeometryParams()
        test_params.ts = base_params.ts.copy()
        
        # Apply the offsets to all airfoil layers
        for layer in test_params.ts:
            layer['y_offset'] = tc['y_offset']
            layer['z_offset'] = tc['z_offset']
        
        # Generate the wing
        output_file = TEST_DIR / f"wing_{tc['name']}.stl"
        try:
            wing_generator.create_geometry_from_array(test_params.ts, AIRFOIL_FILES, str(output_file))
            
            # Load the mesh to calculate penalty
            mesh_wing = trimesh.load(output_file)
            
            # Calculate bounding box penalty
            penalty = calculate_bounding_box_penalty(mesh_wing, BOUNDING_BOX_LIMITS)
            
            # Print the result
            print(f"Test case '{tc['name']}': Penalty = {penalty}")
            print(f"  - Mesh bounds: Min {mesh_wing.bounds[0]}, Max {mesh_wing.bounds[1]}")
            
            # Store result
            results.append({
                "case": tc['name'],
                "penalty": penalty,
                "bounds_min": mesh_wing.bounds[0].tolist(),
                "bounds_max": mesh_wing.bounds[1].tolist(),
                "file": str(output_file)
            })
            
        except Exception as e:
            print(f"Error in test case '{tc['name']}': {e}")
    
    return results

def create_combined_visualization(results):
    """Create a combined mesh with the bounding box and all test wings for visualization"""
    meshes = []
    
    # Add the bounding box (make it wire frame for visibility)
    bbox_mesh = create_bounding_box_mesh()
    # Make it wireframe in some viewers (not supported in all formats)
    bbox_mesh.visual.face_colors = [100, 100, 100, 50]  # Semi-transparent gray
    meshes.append(bbox_mesh)
    
    # Add each test wing mesh with different colors
    colors = [
        [255, 0, 0, 255],    # Red
        [0, 255, 0, 255],    # Green
        [0, 0, 255, 255],    # Blue
        [255, 255, 0, 255],  # Yellow
        [255, 0, 255, 255],  # Magenta
        [0, 255, 255, 255],  # Cyan
        [255, 128, 0, 255]   # Orange
    ]
    
    for i, result in enumerate(results):
        try:
            mesh = trimesh.load(result["file"])
            color_idx = i % len(colors)
            mesh.visual.face_colors = colors[color_idx]
            meshes.append(mesh)
        except Exception as e:
            print(f"Error loading mesh {result['file']}: {e}")
    
    # Create a scene with all meshes
    scene = trimesh.Scene(meshes)
    
    # Export the combined visualization
    output_path = TEST_DIR / "combined_visualization.stl"
    scene.export(output_path)
    print(f"Combined visualization exported to {output_path}")
    
    # Also export as glb format if possible (supports transparency)
    try:
        glb_path = TEST_DIR / "combined_visualization.glb"
        scene.export(glb_path)
        print(f"GLB visualization exported to {glb_path}")
    except Exception as e:
        print(f"Could not export to GLB format: {e}")

def print_bounding_box_details():
    """Print out the bounding box details for reference"""
    print("\n--- Bounding Box Configuration ---")
    print(f"X range: {BOUNDING_BOX_LIMITS['x_min']:.6f} to {BOUNDING_BOX_LIMITS['x_max']:.6f}")
    print(f"Y range: {BOUNDING_BOX_LIMITS['y_min']:.6f} to {BOUNDING_BOX_LIMITS['y_max']:.6f}")
    print(f"Z range: {BOUNDING_BOX_LIMITS['z_min']:.6f} to {BOUNDING_BOX_LIMITS['z_max']:.6f}")
    
    # Calculate dimensions
    x_size = BOUNDING_BOX_LIMITS['x_max'] - BOUNDING_BOX_LIMITS['x_min']
    y_size = BOUNDING_BOX_LIMITS['y_max'] - BOUNDING_BOX_LIMITS['y_min']
    z_size = BOUNDING_BOX_LIMITS['z_max'] - BOUNDING_BOX_LIMITS['z_min']
    
    print(f"Dimensions: X={x_size:.6f}, Y={y_size:.6f}, Z={z_size:.6f}")
    print("---------------------------------\n")

if __name__ == "__main__":
    print("Starting bounding box penalty test...")
    print(f"Results will be saved to: {TEST_DIR}")
    
    # Print bounding box configuration
    print_bounding_box_details()
    
    # Create test wings and check penalties
    results = create_test_wings()
    
    # Create combined visualization
    create_combined_visualization(results)
    
    # Summary
    print("\n--- Test Summary ---")
    for result in results:
        penalty_status = "OUTSIDE BOX" if result["penalty"] == 1 else "INSIDE BOX"
        print(f"{result['case']}: {penalty_status}")
    
    print("\nTest completed! Check the test_results directory for STL files.")
