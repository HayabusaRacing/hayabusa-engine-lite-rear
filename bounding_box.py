import trimesh
import numpy as np
from config import AIRFOIL_FILES, AIRFOIL_WING_CHORD, AIRFOIL_WING_SPAN
from geometry.airfoilLayers import airfoilLayers
from pathlib import Path

def calculate_bounding_box_trimesh(points):
    """
    Calculate the bounding box for a set of 3D points using trimesh.
    
    Args:
        points: List of [x, y, z] coordinates
    
    Returns:
        A trimesh object representing the bounding box
    """
    # Create a Trimesh object from the points
    cloud = trimesh.points.PointCloud(points)
    
    # Get the bounding box as a Trimesh object
    bounding_box = cloud.bounding_box
    return bounding_box

def save_stl(mesh, filename):
    """
    Save a trimesh object as an STL file.
    
    Args:
        mesh: A trimesh object
        filename: Output STL file name
    """
    mesh.export(filename)
    print(f"STL saved as '{filename}'")

def main():
    # Configuration
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    wing_stl_path = output_dir / "wing.stl"
    bounding_box_stl_path = output_dir / "bounding_box.stl"
    
    # Generate the wing
    param_array = [0, 0.0, 0.0, 0.0, 1.0]  # Example parameters
    wing = airfoilLayers(
        density=2,  # Center layer + one optimizable layer
        wing_span=AIRFOIL_WING_SPAN,
        wing_chord=AIRFOIL_WING_CHORD
    )
    wing.create_geometry_from_array(param_array, AIRFOIL_FILES, str(wing_stl_path))
    print(f"Wing STL saved as '{wing_stl_path}'")
    
    # Collect all points from the wing surface
    all_points = []
    for layer in wing.layers:
        all_points.extend(layer.coords)
    
    # Create a Trimesh object for the wing
    wing_mesh = trimesh.Trimesh(vertices=all_points)
    
    # Calculate the bounding box
    bounding_box = calculate_bounding_box_trimesh(all_points)
    print(f"Bounding box dimensions: {bounding_box.extents}")
    
    # Save the bounding box STL
    save_stl(bounding_box, str(bounding_box_stl_path))

if __name__ == "__main__":
    main()