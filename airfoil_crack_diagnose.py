#!/usr/bin/env python3

"""
Leading Edge Crack Diagnostic Test
This script tests different aspects of the airfoil generation to identify the cause of front cracks.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).resolve().parent))

from geometry.airfoilLayers import airfoilLayers, airfoilLayer, geometryParameter
from config import AIRFOIL_FILES

def visualize_single_airfoil(airfoil_file, output_dir="test_outputs"):
    """Visualize a single airfoil profile to check point distribution"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the airfoil
    layer = airfoilLayer(airfoil_file)
    
    # Extract coordinates
    points = np.array(layer.coords)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # Plot the airfoil profile
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(y, z, 'bo-')
    plt.plot(y, z, 'r.')  # Points as red dots
    plt.title(f"Airfoil Profile: {os.path.basename(airfoil_file)}")
    plt.xlabel("Chord direction (Y)")
    plt.ylabel("Height (Z)")
    plt.grid(True)
    plt.axis('equal')
    
    # Zoom in on leading edge
    plt.subplot(2, 1, 2)
    y_min = min(y)
    y_max = y_min + (max(y) - y_min) * 0.2  # 20% of chord from leading edge
    z_mid = np.mean(z)
    z_range = max(z) - min(z)
    plt.plot(y, z, 'bo-')
    plt.plot(y, z, 'r.')  # Points as red dots
    plt.xlim(y_min - 0.01, y_max)
    plt.ylim(z_mid - z_range/2, z_mid + z_range/2)
    plt.title("Leading Edge Detail (Red dots = points)")
    plt.xlabel("Chord direction (Y)")
    plt.ylabel("Height (Z)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{os.path.basename(airfoil_file)}_profile.png"))
    plt.close()
    
    return points

def visualize_wing_leading_edge(airfoil_files, output_dir="test_outputs"):
    """Generate a wing and visualize its leading edge points"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a wing with density=2 (1 fixed center + 1 optimizable layer)
    wing = airfoilLayers(density=2, wing_span=0.065/2, wing_chord=0.02)
    
    # Generate parameters for the wing - now correct for density=2
    param_array = [0, 0.0, 0.0, 0.0, 1.0]  # Just one optimizable layer
    
    # Generate the wing
    stl_file = os.path.join(output_dir, "test_wing.stl")
    wing.create_geometry_from_array(param_array, airfoil_files, stl_file)
    
    # Extract the leading edge points from each layer
    leading_edge_points = []
    for layer in wing.layers:
        # Find leading edge index (minimum Y coordinate)
        y_coords = [p[1] for p in layer.coords]
        le_idx = y_coords.index(min(y_coords))
        leading_edge_points.append(layer.coords[le_idx])
    
    # Visualize in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each layer
    for i, layer in enumerate(wing.layers):
        points = np.array(layer.coords)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=10, label=f"Layer {i}")
    
    # Highlight leading edge points
    le_points = np.array(leading_edge_points)
    ax.scatter(le_points[:, 0], le_points[:, 1], le_points[:, 2], color='red', s=100, marker='o', label="Leading Edges")
    
    ax.set_xlabel('X (Span)')
    ax.set_ylabel('Y (Chord)')
    ax.set_zlabel('Z (Height)')
    ax.set_title('Wing Layers with Leading Edge Highlighted')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "leading_edge_visualization.png"))
    plt.close()
    
    return wing

def check_surface_to_endcap_connection(wing, output_dir="test_outputs"):
    """Examine if there are gaps between the B-spline surface and end caps"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the surface
    surf = wing._create_surface()
    surf.evaluate()
    points = np.array(surf.evalpts)
    
    # Get surface dimensions
    u_count, v_count = wing._get_surface_dimensions(surf, points)
    point_grid = points.reshape(v_count, u_count, 3)
    
    # Get the points along the edges (where end caps connect)
    edge1_points = point_grid[0, :, :]  # First row
    edge2_points = point_grid[-1, :, :]  # Last row
    
    # Get end cap points
    endcap1_points = np.array(wing.layers[0].coords)
    endcap2_points = np.array(wing.layers[-1].coords)
    
    # Plot the connections
    fig = plt.figure(figsize=(15, 10))
    
    # First end cap
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(edge1_points[:, 0], edge1_points[:, 1], edge1_points[:, 2], c='blue', marker='o', label='Surface Edge')
    ax1.scatter(endcap1_points[:, 0], endcap1_points[:, 1], endcap1_points[:, 2], c='red', marker='^', label='End Cap Points')
    ax1.set_title('First End Cap Connection')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    
    # Second end cap
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(edge2_points[:, 0], edge2_points[:, 1], edge2_points[:, 2], c='blue', marker='o', label='Surface Edge')
    ax2.scatter(endcap2_points[:, 0], endcap2_points[:, 1], endcap2_points[:, 2], c='red', marker='^', label='End Cap Points')
    ax2.set_title('Second End Cap Connection')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "endcap_connection_check.png"))
    plt.close()

    # Calculate distances between surface edges and end cap points
    def min_point_distance(edge_points, cap_points):
        min_dists = []
        for cap_pt in cap_points:
            dists = np.linalg.norm(edge_points - cap_pt, axis=1)
            min_dists.append(np.min(dists))
        return min_dists
    
    edge1_dists = min_point_distance(edge1_points, endcap1_points)
    edge2_dists = min_point_distance(edge2_points, endcap2_points)
    
    # Print statistics
    print(f"First end cap - Avg distance: {np.mean(edge1_dists):.6f}, Max: {np.max(edge1_dists):.6f}")
    print(f"Second end cap - Avg distance: {np.mean(edge2_dists):.6f}, Max: {np.max(edge2_dists):.6f}")
    
    return edge1_dists, edge2_dists

if __name__ == "__main__":
    print("Running Leading Edge Crack Diagnostic...")
    
    # Test all airfoil profiles
    for airfoil in AIRFOIL_FILES:
        print(f"Testing airfoil: {airfoil}")
        visualize_single_airfoil(airfoil)
    
    # Test wing with a single airfoil type
    print("Testing wing with single airfoil...")
    wing = visualize_wing_leading_edge(AIRFOIL_FILES[:1])
    
    # Check surface-to-endcap connections
    print("Checking surface-to-endcap connections...")
    check_surface_to_endcap_connection(wing)
    
    print("Diagnostic complete. Check the test_outputs directory for visualizations.")