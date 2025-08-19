import trimesh
import numpy as np
from config import BOUNDING_BOX_LIMITS
from pathlib import Path

def create_bounding_box_mesh(bounding_box_limits):
    """
    Create a trimesh object representing the bounding box.
    
    Args:
        bounding_box_limits: A dictionary defining the bounding box limits
    
    Returns:
        A trimesh object representing the bounding box
    """
    # Extract bounding box limits
    x_min = bounding_box_limits['x_min']
    x_max = bounding_box_limits['x_max']
    y_min = bounding_box_limits['y_min']
    y_max = bounding_box_limits['y_max']
    z_min = bounding_box_limits['z_min']
    z_max = bounding_box_limits['z_max']
    
    # Define the 8 vertices of the bounding box
    vertices = [
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
        [x_min, y_max, z_max],
    ]
    
    # Define the 12 triangles of the bounding box
    faces = [
        [0, 1, 2], [0, 2, 3],  # Bottom face
        [4, 5, 6], [4, 6, 7],  # Top face
        [0, 1, 5], [0, 5, 4],  # Front face
        [2, 3, 7], [2, 7, 6],  # Back face
        [1, 2, 6], [1, 6, 5],  # Right face
        [0, 3, 7], [0, 7, 4],  # Left face
    ]
    
    # Create the mesh
    bounding_box_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return bounding_box_mesh

def save_bounding_box_stl(output_path):
    """
    Generate and save the bounding box STL file.
    
    Args:
        output_path: Path to save the STL file
    """
    # Create the bounding box mesh
    bounding_box_mesh = create_bounding_box_mesh(BOUNDING_BOX_LIMITS)
    
    # Save the STL file
    bounding_box_mesh.export(output_path)
    print(f"Bounding box STL saved as '{output_path}'")

def main():
    # Define the output directory and file
    output_dir = Path("bounding_box_output")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "bounding_box.stl"
    
    # Generate and save the bounding box STL
    save_bounding_box_stl(output_file)

if __name__ == "__main__":
    main()