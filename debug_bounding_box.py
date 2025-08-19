"""
Debugging script for bounding box penalty calculation.
This script tests a single wing against the bounding box with detailed debugging.
"""

import sys
import os
import numpy as np
import trimesh
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from geometry.airfoilLayers import airfoilLayers
from geometry.geometryParams import GeometryParams
from ga.evaluate import calculate_bounding_box_penalty
from config import (BOUNDING_BOX_LIMITS, AIRFOIL_FILES, TMP_DIR,
                   AIRFOIL_DENSITY, AIRFOIL_WING_SPAN, AIRFOIL_WING_CHORD,
                   AIRFOIL_X_CENTER, AIRFOIL_Y_CENTER, AIRFOIL_Z_CENTER,
                   AIRFOIL_SURFACE_DEGREE_U, AIRFOIL_SURFACE_DEGREE_V, AIRFOIL_SAMPLE_RESOLUTION)

# Create test directory if it doesn't exist
DEBUG_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "debug_results"
DEBUG_DIR.mkdir(exist_ok=True)

def generate_wing_with_params(params, output_file):
    """Generate a wing with the given parameters and return the mesh"""
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
    
    wing_generator.create_geometry_from_array(
        params.ts, AIRFOIL_FILES, str(output_file)
    )
    
    mesh_wing = trimesh.load(output_file)
    return mesh_wing

def visualize_wing_and_bounding_box(mesh_wing, ax=None):
    """Visualize the wing and bounding box for debugging"""
    if ax is None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
    
    # Extract wing bounds
    wing_bounds = mesh_wing.bounds
    
    # Plot wing bounding box as a wireframe
    wing_min = wing_bounds[0]
    wing_max = wing_bounds[1]
    
    # Create points for the wing's bounding box
    wing_x = [wing_min[0], wing_max[0]]
    wing_y = [wing_min[1], wing_max[1]]
    wing_z = [wing_min[2], wing_max[2]]
    
    # Create the wireframe for the wing's bounding box
    for i in range(2):
        for j in range(2):
            ax.plot([wing_x[0], wing_x[1]], [wing_y[i], wing_y[i]], [wing_z[j], wing_z[j]], 'b-', linewidth=2)
            ax.plot([wing_x[i], wing_x[i]], [wing_y[0], wing_y[1]], [wing_z[j], wing_z[j]], 'b-', linewidth=2)
            ax.plot([wing_x[i], wing_x[i]], [wing_y[j], wing_y[j]], [wing_z[0], wing_z[1]], 'b-', linewidth=2)
    
    # Extract limits from config
    bb_min = [BOUNDING_BOX_LIMITS['x_min'], BOUNDING_BOX_LIMITS['y_min'], BOUNDING_BOX_LIMITS['z_min']]
    bb_max = [BOUNDING_BOX_LIMITS['x_max'], BOUNDING_BOX_LIMITS['y_max'], BOUNDING_BOX_LIMITS['z_max']]
    
    # Create points for the configured bounding box
    bb_x = [bb_min[0], bb_max[0]]
    bb_y = [bb_min[1], bb_max[1]]
    bb_z = [bb_min[2], bb_max[2]]
    
    # Create the wireframe for the configured bounding box
    for i in range(2):
        for j in range(2):
            ax.plot([bb_x[0], bb_x[1]], [bb_y[i], bb_y[i]], [bb_z[j], bb_z[j]], 'r-', linewidth=1)
            ax.plot([bb_x[i], bb_x[i]], [bb_y[0], bb_y[1]], [bb_z[j], bb_z[j]], 'r-', linewidth=1)
            ax.plot([bb_x[i], bb_x[i]], [bb_y[j], bb_y[j]], [bb_z[0], bb_z[1]], 'r-', linewidth=1)
    
    # Plot wing vertices
    wing_vertices = mesh_wing.vertices
    ax.scatter(wing_vertices[:, 0], wing_vertices[:, 1], wing_vertices[:, 2], 
               c='g', marker='.', alpha=0.1, s=1)
    
    # Set labels and title
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Wing and Bounding Box Visualization')
    
    # Add a legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='b', lw=2, label='Wing Bounds'),
        Line2D([0], [0], color='r', lw=1, label='Required Bounding Box'),
        Line2D([0], [0], marker='.', color='g', lw=0, label='Wing Vertices', 
               markersize=8, alpha=0.5)
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    return ax

def debug_bounding_box_test(params_list, labels):
    """Run detailed debugging tests for the given parameters"""
    results = []
    
    fig = plt.figure(figsize=(20, 15))
    
    for i, (params, label) in enumerate(zip(params_list, labels)):
        print(f"\n--- Testing {label} ---")
        
        # Generate the wing
        output_file = DEBUG_DIR / f"wing_{label}.stl"
        mesh_wing = generate_wing_with_params(params, output_file)
        
        # Calculate bounds and bounding box penalty
        bounds = mesh_wing.bounds
        min_coords = bounds[0]
        max_coords = bounds[1]
        
        # Print detailed information
        print(f"Wing bounds:")
        print(f"  X: {min_coords[0]:.6f} to {max_coords[0]:.6f}")
        print(f"  Y: {min_coords[1]:.6f} to {max_coords[1]:.6f}")
        print(f"  Z: {min_coords[2]:.6f} to {max_coords[2]:.6f}")
        
        print(f"Bounding box limits:")
        print(f"  X: {BOUNDING_BOX_LIMITS['x_min']:.6f} to {BOUNDING_BOX_LIMITS['x_max']:.6f}")
        print(f"  Y: {BOUNDING_BOX_LIMITS['y_min']:.6f} to {BOUNDING_BOX_LIMITS['y_max']:.6f}")
        print(f"  Z: {BOUNDING_BOX_LIMITS['z_min']:.6f} to {BOUNDING_BOX_LIMITS['z_max']:.6f}")
        
        # Check detailed bounds violations
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
        
        penalty = calculate_bounding_box_penalty(mesh_wing, BOUNDING_BOX_LIMITS)
        
        # Print result
        if violations:
            print("Violations found:")
            for v in violations:
                print(f"  - {v}")
            print(f"Penalty: {penalty}")
        else:
            print(f"No violations found. Penalty: {penalty}")
        
        # Store result
        result = {
            "label": label,
            "penalty": penalty,
            "violations": violations,
            "bounds_min": min_coords.tolist(),
            "bounds_max": max_coords.tolist(),
            "file": str(output_file)
        }
        results.append(result)
        
        # Add visualization
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        visualize_wing_and_bounding_box(mesh_wing, ax)
        ax.set_title(f"{label} - Penalty: {penalty}")
    
    plt.tight_layout()
    plt.savefig(DEBUG_DIR / "debug_visualization.png", dpi=300)
    print(f"Debug visualization saved to {DEBUG_DIR / 'debug_visualization.png'}")
    
    return results

def main():
    print(f"Starting bounding box debug test...")
    print(f"Results will be saved to: {DEBUG_DIR}")
    
    # Create test parameters
    # 1. A wing with default parameters
    default_params = GeometryParams()
    default_params.ts = [
        {'wing_type_idx': 0, 'pitch_angle': 0, 'y_offset': 0, 'z_offset': 0, 'scale': 1.0},
        {'wing_type_idx': 1, 'pitch_angle': 0, 'y_offset': 0, 'z_offset': 0, 'scale': 1.0},
        {'wing_type_idx': 0, 'pitch_angle': 0, 'y_offset': 0, 'z_offset': 0, 'scale': 1.0}
    ]
    
    # 2. A wing with Y coordinates modified to be inside the box
    inside_y_params = GeometryParams()
    inside_y_params.ts = [
        {'wing_type_idx': 0, 'pitch_angle': 0, 'y_offset': 0, 'z_offset': 0, 'scale': 1.0},
        {'wing_type_idx': 1, 'pitch_angle': 0, 'y_offset': 0, 'z_offset': 0, 'scale': 1.0},
        {'wing_type_idx': 0, 'pitch_angle': 0, 'y_offset': 0, 'z_offset': 0, 'scale': 1.0}
    ]
    # Apply an offset to ensure it's centered in the Y range of the bounding box
    y_center = (BOUNDING_BOX_LIMITS['y_min'] + BOUNDING_BOX_LIMITS['y_max']) / 2
    y_offset = y_center - AIRFOIL_Y_CENTER
    for layer in inside_y_params.ts:
        layer['y_offset'] = y_offset
    
    # 3. A wing with Z coordinates modified to be inside the box
    inside_z_params = GeometryParams()
    inside_z_params.ts = [
        {'wing_type_idx': 0, 'pitch_angle': 0, 'y_offset': 0, 'z_offset': 0, 'scale': 1.0},
        {'wing_type_idx': 1, 'pitch_angle': 0, 'y_offset': 0, 'z_offset': 0, 'scale': 1.0},
        {'wing_type_idx': 0, 'pitch_angle': 0, 'y_offset': 0, 'z_offset': 0, 'scale': 1.0}
    ]
    # Apply an offset to ensure it's centered in the Z range of the bounding box
    z_center = (BOUNDING_BOX_LIMITS['z_min'] + BOUNDING_BOX_LIMITS['z_max']) / 2
    z_offset = z_center - AIRFOIL_Z_CENTER
    for layer in inside_z_params.ts:
        layer['z_offset'] = z_offset
    
    # 4. A wing with both Y and Z coordinates modified to be inside the box
    inside_both_params = GeometryParams()
    inside_both_params.ts = [
        {'wing_type_idx': 0, 'pitch_angle': 0, 'y_offset': 0, 'z_offset': 0, 'scale': 1.0},
        {'wing_type_idx': 1, 'pitch_angle': 0, 'y_offset': 0, 'z_offset': 0, 'scale': 1.0},
        {'wing_type_idx': 0, 'pitch_angle': 0, 'y_offset': 0, 'z_offset': 0, 'scale': 1.0}
    ]
    # Apply offsets for both y and z
    for layer in inside_both_params.ts:
        layer['y_offset'] = y_offset
        layer['z_offset'] = z_offset
    
    # Run debug tests
    results = debug_bounding_box_test(
        [default_params, inside_y_params, inside_z_params, inside_both_params],
        ["Default", "Y_Centered", "Z_Centered", "YZ_Centered"]
    )
    
    print("\n--- Test Summary ---")
    for result in results:
        penalty_status = "OUTSIDE BOX" if result["penalty"] == 1 else "INSIDE BOX"
        print(f"{result['label']}: {penalty_status}")
        if result["violations"]:
            print(f"  Violations: {len(result['violations'])}")
    
    print("\nDebug test completed! Check the debug_results directory for STL files and visualizations.")

if __name__ == "__main__":
    main()
