"""
Simple bounding box test script that skips all PyFOAM processing.
This script only tests the wing geometry generation and bounding box penalty calculation.
"""

import os
import sys
import numpy as np
import trimesh
from pathlib import Path

# Import only the necessary modules without PyFOAM dependencies
from geometry.airfoilLayers import airfoilLayers
from geometry.geometryParams import GeometryParams
from config import (BOUNDING_BOX_LIMITS, AIRFOIL_FILES, TMP_DIR,
                   AIRFOIL_DENSITY, AIRFOIL_WING_SPAN, AIRFOIL_WING_CHORD,
                   AIRFOIL_X_CENTER, AIRFOIL_Y_CENTER, AIRFOIL_Z_CENTER,
                   AIRFOIL_SURFACE_DEGREE_U, AIRFOIL_SURFACE_DEGREE_V, AIRFOIL_SAMPLE_RESOLUTION)

# Copy the function directly to avoid importing evaluate.py which has PyFOAM dependencies
def calculate_bounding_box_penalty(mesh, bounding_box_limits):
    """
    Calculate a binary penalty if the mesh is outside the defined bounding box.
    
    Args:
        mesh: The trimesh object of the wing
        bounding_box_limits: A dictionary defining the bounding box limits
    
    Returns:
        1 if the mesh is outside the bounding box, 0 otherwise
    """
    bounds = mesh.bounds
    min_coords = bounds[0]
    max_coords = bounds[1]
    
    # Handle possibly inverted y-bounds in the config
    y_min_actual = min(bounding_box_limits['y_min'], bounding_box_limits['y_max'])
    y_max_actual = max(bounding_box_limits['y_min'], bounding_box_limits['y_max'])
    
    # Detailed bounds checking with logging
    violations = []
    
    # Check X bounds
    if min_coords[0] < bounding_box_limits['x_min']:
        violations.append(f"X min violation: {min_coords[0]:.6f} < {bounding_box_limits['x_min']:.6f}")
    if max_coords[0] > bounding_box_limits['x_max']:
        violations.append(f"X max violation: {max_coords[0]:.6f} > {bounding_box_limits['x_max']:.6f}")
    
    # Check Y bounds with corrected min/max values
    if min_coords[1] < y_min_actual:
        violations.append(f"Y min violation: {min_coords[1]:.6f} < {y_min_actual:.6f}")
    if max_coords[1] > y_max_actual:
        violations.append(f"Y max violation: {max_coords[1]:.6f} > {y_max_actual:.6f}")
    
    # Check Z bounds
    if min_coords[2] < bounding_box_limits['z_min']:
        violations.append(f"Z min violation: {min_coords[2]:.6f} < {bounding_box_limits['z_min']:.6f}")
    if max_coords[2] > bounding_box_limits['z_max']:
        violations.append(f"Z max violation: {max_coords[2]:.6f} > {bounding_box_limits['z_max']:.6f}")
    
    # Check if the mesh is outside the bounding box
    if violations:
        print(f"BOUNDING BOX VIOLATIONS:")
        for violation in violations:
            print(f"  - {violation}")
        return 1  # Penalty for being outside the bounding box
        
    print(f"BOUNDING BOX: All constraints satisfied")
    return 0  # No penalty

# Create test directory if it doesn't exist
TEST_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "bb_test_results"
TEST_DIR.mkdir(exist_ok=True)

def create_bounding_box_stl():
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
    
    # Export as STL
    bbox_path = TEST_DIR / "bounding_box.stl"
    bbox_mesh.export(bbox_path)
    print(f"Bounding box STL exported to {bbox_path}")
    
    return bbox_path

def generate_test_wing(params, name):
    """Generate a test wing with the given parameters and return the mesh"""
    try:
        # Create wing generator
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
        
        # Generate wing STL
        output_file = TEST_DIR / f"wing_{name}.stl"
        wing_generator.create_geometry_from_array(params.ts, AIRFOIL_FILES, str(output_file))
        
        # Load the mesh
        mesh_wing = trimesh.load(output_file)
        
        return mesh_wing, output_file
    
    except Exception as e:
        print(f"Error generating wing '{name}': {e}")
        return None, None

def test_wing_with_offsets(base_name, y_offset=0.0, z_offset=0.0, pitch=0.0, scale=1.0):
    """Test a wing with specific offsets and check bounding box penalty"""
    # Create parameters with ts data - need to use the expected 10 parameters format
    # Format is [wing_type_idx, pitch_angle, y_offset, z_offset, scale] for each optimizable layer
    ts = [
        0, pitch, y_offset, z_offset, scale,  # First optimizable layer
        1, pitch, y_offset, z_offset, scale   # Second optimizable layer
    ]
    params = GeometryParams(ts=ts)
    
    # Generate the wing
    mesh_wing, stl_path = generate_test_wing(params, base_name)
    
    if mesh_wing is None:
        return {
            "name": base_name,
            "success": False,
            "error": "Failed to generate wing"
        }
    
    # Get the wing bounds
    bounds = mesh_wing.bounds
    min_coords = bounds[0]
    max_coords = bounds[1]
    
    # Check bounding box penalty
    penalty = calculate_bounding_box_penalty(mesh_wing, BOUNDING_BOX_LIMITS)
    
    return {
        "name": base_name,
        "success": True,
        "penalty": penalty,
        "bounds": {
            "min": min_coords.tolist(),
            "max": max_coords.tolist()
        },
        "stl_path": str(stl_path)
    }

def run_bounding_box_tests():
    """Run a series of tests with different wing configurations"""
    print(f"\n--- Bounding Box Test ---")
    print(f"Bounding box limits:")
    print(f"  X: {BOUNDING_BOX_LIMITS['x_min']:.6f} to {BOUNDING_BOX_LIMITS['x_max']:.6f}")
    print(f"  Y: {BOUNDING_BOX_LIMITS['y_min']:.6f} to {BOUNDING_BOX_LIMITS['y_max']:.6f}")
    print(f"  Z: {BOUNDING_BOX_LIMITS['z_min']:.6f} to {BOUNDING_BOX_LIMITS['z_max']:.6f}")
    
    # Calculate center points of the bounding box
    y_center = (BOUNDING_BOX_LIMITS['y_min'] + BOUNDING_BOX_LIMITS['y_max']) / 2
    z_center = (BOUNDING_BOX_LIMITS['z_min'] + BOUNDING_BOX_LIMITS['z_max']) / 2
    
    # Calculate required offsets to center the wing in the bounding box
    y_offset_to_center = y_center - AIRFOIL_Y_CENTER
    z_offset_to_center = z_center - AIRFOIL_Z_CENTER
    
    print(f"\nDefault wing centers:")
    print(f"  Y center: {AIRFOIL_Y_CENTER:.6f}")
    print(f"  Z center: {AIRFOIL_Z_CENTER:.6f}")
    
    print(f"\nOffsets to center in bounding box:")
    print(f"  Y offset: {y_offset_to_center:.6f}")
    print(f"  Z offset: {z_offset_to_center:.6f}")
    
    # NOTE: There seems to be an issue in the config - y_min should be smaller than y_max
    # But in the config it shows y_min = -0.0856 and y_max = -0.1256
    # Let's correct this for the test to ensure logical consistency
    actual_y_min = min(BOUNDING_BOX_LIMITS['y_min'], BOUNDING_BOX_LIMITS['y_max'])
    actual_y_max = max(BOUNDING_BOX_LIMITS['y_min'], BOUNDING_BOX_LIMITS['y_max'])
    
    print(f"\nNOTE: Y bounds in config appear to be inverted.")
    print(f"Using corrected values: y_min={actual_y_min:.6f}, y_max={actual_y_max:.6f}")
    
    # Test cases to run
    test_cases = [
        # Test default position with no offsets from config
        {"name": "default", "y": 0.0, "z": 0.0, "pitch": 0.0, "scale": 1.0},
        
        # Test centering in bounding box
        {"name": "centered_y", "y": y_offset_to_center, "z": 0.0, "pitch": 0.0, "scale": 1.0},
        {"name": "centered_z", "y": 0.0, "z": z_offset_to_center, "pitch": 0.0, "scale": 1.0},
        {"name": "centered_yz", "y": y_offset_to_center, "z": z_offset_to_center, "pitch": 0.0, "scale": 1.0},
        
        # Test actual positions at each edge of the bounding box
        # Note: We can't directly test x-bounds since x_offset is not a parameter in GeometryParams
        {"name": "actual_y_min_edge", "y": actual_y_min - AIRFOIL_Y_CENTER + 0.001, "z": 0.0, "pitch": 0.0, "scale": 1.0},
        {"name": "actual_y_max_edge", "y": actual_y_max - AIRFOIL_Y_CENTER - 0.001, "z": 0.0, "pitch": 0.0, "scale": 1.0},
        {"name": "actual_z_min_edge", "y": 0.0, "z": BOUNDING_BOX_LIMITS['z_min'] - AIRFOIL_Z_CENTER + 0.001, "pitch": 0.0, "scale": 1.0},
        {"name": "actual_z_max_edge", "y": 0.0, "z": BOUNDING_BOX_LIMITS['z_max'] - AIRFOIL_Z_CENTER - 0.001, "pitch": 0.0, "scale": 1.0},
        
        # Test violation cases at each edge
        # Note: x-bounds tests removed since x_offset is not available in GeometryParams
        {"name": "actual_y_min_violation", "y": actual_y_min - AIRFOIL_Y_CENTER - 0.01, "z": 0.0, "pitch": 0.0, "scale": 1.0},
        {"name": "actual_y_max_violation", "y": actual_y_max - AIRFOIL_Y_CENTER + 0.01, "z": 0.0, "pitch": 0.0, "scale": 1.0},
        {"name": "actual_z_min_violation", "y": 0.0, "z": BOUNDING_BOX_LIMITS['z_min'] - AIRFOIL_Z_CENTER - 0.01, "pitch": 0.0, "scale": 1.0},
        {"name": "actual_z_max_violation", "y": 0.0, "z": BOUNDING_BOX_LIMITS['z_max'] - AIRFOIL_Z_CENTER + 0.01, "pitch": 0.0, "scale": 1.0},
    ]
    
    results = []
    
    # Run all test cases
    print("\nRunning tests...")
    for tc in test_cases:
        print(f"\nTesting: {tc['name']}")
        result = test_wing_with_offsets(
            tc['name'], 
            y_offset=tc['y'], 
            z_offset=tc['z'],
            pitch=tc.get('pitch', 0.0), 
            scale=tc.get('scale', 1.0)
        )
        results.append(result)
        
        if result['success']:
            print(f"  Penalty: {result['penalty']}")
            print(f"  Min bounds: {result['bounds']['min']}")
            print(f"  Max bounds: {result['bounds']['max']}")
            print(f"  STL saved: {result['stl_path']}")
        else:
            print(f"  Failed: {result.get('error', 'Unknown error')}")
    
    # Create bounding box visualization
    bbox_path = create_bounding_box_stl()
    
    return results

def export_report(results):
    """Export a summary report of the tests"""
    report_path = TEST_DIR / "test_report.txt"
    
    with open(report_path, "w") as f:
        f.write("Bounding Box Test Report\n")
        f.write("======================\n\n")
        
        f.write("Bounding Box Limits:\n")
        f.write(f"  X: {BOUNDING_BOX_LIMITS['x_min']:.6f} to {BOUNDING_BOX_LIMITS['x_max']:.6f}\n")
        f.write(f"  Y: {BOUNDING_BOX_LIMITS['y_min']:.6f} to {BOUNDING_BOX_LIMITS['y_max']:.6f}\n")
        f.write(f"  Z: {BOUNDING_BOX_LIMITS['z_min']:.6f} to {BOUNDING_BOX_LIMITS['z_max']:.6f}\n\n")
        
        f.write("Test Results:\n")
        for result in results:
            if result['success']:
                penalty_status = "OUTSIDE BOX" if result['penalty'] == 1 else "INSIDE BOX"
                f.write(f"\n{result['name']}: {penalty_status}\n")
                f.write(f"  Min bounds: {result['bounds']['min']}\n")
                f.write(f"  Max bounds: {result['bounds']['max']}\n")
                
                # Check which dimension caused the violation
                if result['penalty'] == 1:
                    min_coords = result['bounds']['min']
                    max_coords = result['bounds']['max']
                    violations = []
                    
                    if min_coords[0] < BOUNDING_BOX_LIMITS['x_min']:
                        violations.append(f"X min violation: {min_coords[0]:.6f} < {BOUNDING_BOX_LIMITS['x_min']:.6f}")
                    if max_coords[0] > BOUNDING_BOX_LIMITS['x_max']:
                        violations.append(f"X max violation: {max_coords[0]:.6f} > {BOUNDING_BOX_LIMITS['x_max']:.6f}")
                    if min_coords[1] < BOUNDING_BOX_LIMITS['y_min']:
                        violations.append(f"Y min violation: {min_coords[1]:.6f} < {BOUNDING_BOX_LIMITS['y_min']:.6f}")
                    if max_coords[1] > BOUNDING_BOX_LIMITS['y_max']:
                        violations.append(f"Y max violation: {max_coords[1]:.6f} > {BOUNDING_BOX_LIMITS['y_max']:.6f}")
                    if min_coords[2] < BOUNDING_BOX_LIMITS['z_min']:
                        violations.append(f"Z min violation: {min_coords[2]:.6f} < {BOUNDING_BOX_LIMITS['z_min']:.6f}")
                    if max_coords[2] > BOUNDING_BOX_LIMITS['z_max']:
                        violations.append(f"Z max violation: {max_coords[2]:.6f} > {BOUNDING_BOX_LIMITS['z_max']:.6f}")
                    
                    f.write("  Violations:\n")
                    for v in violations:
                        f.write(f"    - {v}\n")
            else:
                f.write(f"\n{result['name']}: FAILED\n")
                f.write(f"  Error: {result.get('error', 'Unknown error')}\n")
    
    print(f"\nTest report exported to {report_path}")

if __name__ == "__main__":
    print("Starting simple bounding box test (no PyFOAM)...")
    print(f"Results will be saved to: {TEST_DIR}")
    
    # Run the tests
    results = run_bounding_box_tests()
    
    # Export a report
    export_report(results)
    
    # Print summary
    print("\n--- Test Summary ---")
    successful_tests = [r for r in results if r['success']]
    failed_tests = [r for r in results if not r['success']]
    
    inside_box = [r for r in successful_tests if r['penalty'] == 0]
    outside_box = [r for r in successful_tests if r['penalty'] == 1]
    
    print(f"Total tests: {len(results)}")
    print(f"Successful: {len(successful_tests)}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Inside box: {len(inside_box)}")
    print(f"Outside box: {len(outside_box)}")
    
    print("\nTest completed!")
