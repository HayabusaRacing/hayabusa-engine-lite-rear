#!/usr/bin/env python3
"""
Production Wing Generator for Hayabusa Racing
Uses the same logic as ga/evaluate.py for consistency
"""

import sys
from pathlib import Path
import trimesh

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import (
    PROJECT_ROOT, AIRFOIL_FILES, AIRFOIL_DENSITY,
    AIRFOIL_WING_SPAN, AIRFOIL_WING_CHORD, AIRFOIL_Y_CENTER,
    AIRFOIL_X_CENTER, AIRFOIL_Z_CENTER, AIRFOIL_SURFACE_DEGREE_U,
    AIRFOIL_SURFACE_DEGREE_V, AIRFOIL_SAMPLE_RESOLUTION
)
from geometry.airfoilLayers import airfoilLayers

def generate_wing_and_combine():
    """
    Generate wing using airfoilLayers and combine with main body
    Uses the same logic as ga/evaluate.py
    """
    print("=== Wing Generator (Production) ===")
    print()
    
    # Test parameters for 2 outer layers (10 parameters total)
    test_params = [
        0, 0.0, 0.005, 0.005, 1,  # Layer 1: wing_type, pitch, y_offset, z_offset, scale
        0, 0.0, 0.010, 0.010, 1   # Layer 2: wing_type, pitch, y_offset, z_offset, scale
    ]
    
    print(f"Using test parameters: {test_params}")
    print()
    
    try:
        # Step 1: Create wing generator with full config (same as ga/evaluate.py)
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
        
        # Step 2: Generate wing STL
        wing_filename = "wing.stl"
        wing_generator.create_geometry_from_array(
            test_params, AIRFOIL_FILES, wing_filename
        )
        print(f"✅ Wing generated: {wing_filename}")
        
        # Step 3: Load meshes using trimesh (same as ga/evaluate.py)
        mesh_wing = trimesh.load(wing_filename)
        mesh_body = trimesh.load(PROJECT_ROOT / "baseCase" / "constant" / "triSurface" / "mainBodyNoWing.stl")
        
        # Step 4: Combine meshes
        combined = trimesh.util.concatenate([mesh_body, mesh_wing])
        
        # Step 5: Export combined aircraft
        output_path = "complete_aircraft.stl"
        combined.export(output_path)
        
        print(f"✅ Complete aircraft saved: {output_path}")
        print(f"  - Wing triangles: {len(mesh_wing.faces)}")
        print(f"  - Body triangles: {len(mesh_body.faces)}")
        print(f"  - Total triangles: {len(combined.faces)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """
    Main function to generate wing and combine with main body
    """
    success = generate_wing_and_combine()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)