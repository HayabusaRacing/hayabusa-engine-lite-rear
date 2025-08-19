"""
Visualize the bounding box issue by creating STL files for:
1. The wing_centered_z model
2. The wing's actual bounding box (min/max)
3. The configured bounding box limits
"""

import os
import sys
import numpy as np
import trimesh
from pathlib import Path

from config import BOUNDING_BOX_LIMITS

# Location of generated STL files
TEST_DIR = Path("bb_test_results")
WING_NAME = "wing_centered_z"

def create_box_mesh(min_coords, max_coords, name):
    """Create a mesh representing a box with the given min/max coordinates"""
    # Extract bounds
    x_min, y_min, z_min = min_coords
    x_max, y_max, z_max = max_coords
    
    # Create box vertices (8 corners)
    vertices = np.array([
        [x_min, y_min, z_min],  # 0
        [x_max, y_min, z_min],  # 1
        [x_max, y_max, z_min],  # 2
        [x_min, y_max, z_min],  # 3
        [x_min, y_min, z_max],  # 4
        [x_max, y_min, z_max],  # 5
        [x_max, y_max, z_max],  # 6
        [x_min, y_max, z_max]   # 7
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
    box_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Export as STL
    output_path = TEST_DIR / f"{name}.stl"
    box_mesh.export(output_path)
    print(f"Box mesh exported to {output_path}")
    
    return box_mesh, output_path

def create_wireframe_box(min_coords, max_coords, name):
    """Create a wireframe box for better visualization"""
    x_min, y_min, z_min = min_coords
    x_max, y_max, z_max = max_coords
    
    # Define 8 corners of the box
    corners = np.array([
        [x_min, y_min, z_min],  # 0
        [x_max, y_min, z_min],  # 1
        [x_max, y_max, z_min],  # 2
        [x_min, y_max, z_min],  # 3
        [x_min, y_min, z_max],  # 4
        [x_max, y_min, z_max],  # 5
        [x_max, y_max, z_max],  # 6
        [x_min, y_max, z_max]   # 7
    ])
    
    # Define 12 edges of the box (pairs of corner indices)
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting edges
    ]
    
    # Create cylinders for each edge to represent the wireframe
    cylinders = []
    radius = 0.0005  # Small radius for the wireframe
    
    for edge in edges:
        start, end = corners[edge[0]], corners[edge[1]]
        direction = end - start
        height = np.linalg.norm(direction)
        
        if height > 0:
            # Create a cylinder along the z-axis
            cylinder = trimesh.creation.cylinder(radius=radius, height=height)
            
            # Calculate rotation to align with the edge
            if np.allclose(direction / height, [0, 0, 1]):
                # Already aligned with z-axis
                rotation = np.eye(3)
            else:
                # Align with the edge direction
                z_axis = np.array([0, 0, 1])
                direction_unit = direction / height
                axis = np.cross(z_axis, direction_unit)
                if np.linalg.norm(axis) < 1e-6:
                    # Special case for parallel vectors
                    if np.dot(z_axis, direction_unit) > 0:
                        rotation = np.eye(3)
                    else:
                        # Opposite direction, rotate 180 degrees around x-axis
                        rotation = np.array([
                            [1, 0, 0],
                            [0, -1, 0],
                            [0, 0, -1]
                        ])
                else:
                    # General case
                    axis = axis / np.linalg.norm(axis)
                    angle = np.arccos(np.dot(z_axis, direction_unit))
                    
                    # Rodrigues' rotation formula
                    K = np.array([
                        [0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0]
                    ])
                    rotation = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            
            # Apply rotation
            cylinder.apply_transform(
                np.vstack([np.hstack([rotation, np.array([[0], [0], [0]])]), [0, 0, 0, 1]])
            )
            
            # Translate to start position
            translation = np.eye(4)
            translation[:3, 3] = start + direction / 2
            cylinder.apply_transform(translation)
            
            cylinders.append(cylinder)
    
    # Combine all cylinders
    wireframe = trimesh.util.concatenate(cylinders)
    
    # Export as STL
    output_path = TEST_DIR / f"{name}_wireframe.stl"
    wireframe.export(output_path)
    print(f"Wireframe box exported to {output_path}")
    
    return wireframe, output_path

def main():
    # Load the wing mesh
    wing_stl_path = TEST_DIR / f"{WING_NAME}.stl"
    if not wing_stl_path.exists():
        print(f"Error: Wing STL file {wing_stl_path} not found!")
        return
        
    wing_mesh = trimesh.load(wing_stl_path)
    
    # Get the wing's actual bounding box
    wing_bounds = wing_mesh.bounds
    wing_min = wing_bounds[0]
    wing_max = wing_bounds[1]
    
    print("\nWing Dimensions:")
    print(f"  Min coords: {wing_min}")
    print(f"  Max coords: {wing_max}")
    print(f"  Dimensions: {wing_max - wing_min}")
    
    # Create a bounding box for the wing
    wing_box, wing_box_path = create_box_mesh(wing_min, wing_max, f"{WING_NAME}_box")
    
    # Create a wireframe box for the wing
    wing_wireframe, wing_wireframe_path = create_wireframe_box(wing_min, wing_max, f"{WING_NAME}_box")
    
    # Get the configured bounding box limits
    config_min = [BOUNDING_BOX_LIMITS['x_min'], BOUNDING_BOX_LIMITS['y_min'], BOUNDING_BOX_LIMITS['z_min']]
    config_max = [BOUNDING_BOX_LIMITS['x_max'], BOUNDING_BOX_LIMITS['y_max'], BOUNDING_BOX_LIMITS['z_max']]
    
    print("\nConfigured Bounding Box Limits:")
    print(f"  X: {config_min[0]} to {config_max[0]}")
    print(f"  Y: {config_min[1]} to {config_max[1]}")
    print(f"  Z: {config_min[2]} to {config_max[2]}")
    
    # Create a bounding box for the configured limits
    config_box, config_box_path = create_box_mesh(config_min, config_max, "config_box")
    
    # Create a wireframe box for the configured limits
    config_wireframe, config_wireframe_path = create_wireframe_box(config_min, config_max, "config_box")
    
    # Fix the inverted Y coordinates in the config
    corrected_min = config_min.copy()
    corrected_max = config_max.copy()
    corrected_min[1], corrected_max[1] = min(config_min[1], config_max[1]), max(config_min[1], config_max[1])
    
    print("\nCorrected Bounding Box Limits (Y fixed):")
    print(f"  X: {corrected_min[0]} to {corrected_max[0]}")
    print(f"  Y: {corrected_min[1]} to {corrected_max[1]}")
    print(f"  Z: {corrected_min[2]} to {corrected_max[2]}")
    
    # Create a bounding box for the corrected limits
    corrected_box, corrected_box_path = create_box_mesh(corrected_min, corrected_max, "corrected_box")
    
    # Create a wireframe box for the corrected limits
    corrected_wireframe, corrected_wireframe_path = create_wireframe_box(corrected_min, corrected_max, "corrected_box")
    
    # Check for violations with corrected bounding box
    violations = []
    
    # Check X bounds
    if wing_min[0] < corrected_min[0]:
        violations.append(f"X min violation: {wing_min[0]:.6f} < {corrected_min[0]:.6f}")
    if wing_max[0] > corrected_max[0]:
        violations.append(f"X max violation: {wing_max[0]:.6f} > {corrected_max[0]:.6f}")
    
    # Check Y bounds
    if wing_min[1] < corrected_min[1]:
        violations.append(f"Y min violation: {wing_min[1]:.6f} < {corrected_min[1]:.6f}")
    if wing_max[1] > corrected_max[1]:
        violations.append(f"Y max violation: {wing_max[1]:.6f} > {corrected_max[1]:.6f}")
    
    # Check Z bounds
    if wing_min[2] < corrected_min[2]:
        violations.append(f"Z min violation: {wing_min[2]:.6f} < {corrected_min[2]:.6f}")
    if wing_max[2] > corrected_max[2]:
        violations.append(f"Z max violation: {wing_max[2]:.6f} > {corrected_max[2]:.6f}")
    
    print("\nViolations with original config box:")
    print("  - Y min violation: {:.6f} < {:.6f}".format(wing_min[1], config_min[1]))
    print("  - Y max violation: {:.6f} > {:.6f}".format(wing_max[1], config_max[1]))
    
    print("\nViolations with corrected bounds:")
    if violations:
        for v in violations:
            print(f"  - {v}")
    else:
        print("  No violations with corrected bounds!")
    
    print("\nFiles created for visualization:")
    print(f"  1. Original wing STL: {wing_stl_path}")
    print(f"  2. Wing's bounding box: {wing_box_path}")
    print(f"  3. Wing's wireframe box: {wing_wireframe_path}")
    print(f"  4. Config bounding box: {config_box_path}")
    print(f"  5. Config wireframe box: {config_wireframe_path}")
    print(f"  6. Corrected bounding box: {corrected_box_path}")
    print(f"  7. Corrected wireframe box: {corrected_wireframe_path}")
    
    print("\nThe issue is that the Y limits in the config are inverted:")
    print(f"  y_min: {BOUNDING_BOX_LIMITS['y_min']} is larger than y_max: {BOUNDING_BOX_LIMITS['y_max']}")
    print("  This causes both lower and upper bound checks to fail.")

if __name__ == "__main__":
    main()
