#!/usr/bin/env python3

"""
Test script to verify that rotation is now based on layer offset
"""

import sys
sys.path.append('.')

from geometry.airfoilLayers import airfoilLayers, geometryParameter
from config import *
import numpy as np

def test_rotation_with_offset():
    print("Testing rotation based on layer offset...")
    
    # Create a wing with different offsets for each layer
    wing = airfoilLayers(
        density=AIRFOIL_DENSITY,
        y_center=AIRFOIL_Y_CENTER,
        x_center=AIRFOIL_X_CENTER,
        z_center=AIRFOIL_Z_CENTER,
        wing_span=AIRFOIL_WING_SPAN,
        wing_chord=AIRFOIL_WING_CHORD
    )
    
    # Create parameters with different offsets and pitch angles
    parameters = [
        # Center airfoil (fixed)
        geometryParameter(AIRFOIL_FILES[0], 0.0, 0.0, 0.0, 1.0),
        # Layer 1 with offset and pitch
        geometryParameter(AIRFOIL_FILES[0], 10.0, 0.01, 0.005, 1.0),
        # Layer 2 with different offset and pitch  
        geometryParameter(AIRFOIL_FILES[0], -15.0, -0.02, 0.01, 0.8),
    ]
    
    wing.read_parameters(parameters)
    
    print(f"Created wing with {len(wing.layers)} layers")
    
    # Check that each layer has the expected offsets
    for i, layer in enumerate(wing.layers):
        print(f"Layer {i}:")
        print(f"  x_offset: {layer.x_offset:.4f}")
        print(f"  y_offset: {layer.y_offset:.4f}")
        print(f"  z_offset: {layer.z_offset:.4f}")
        print(f"  First coord: [{layer.coords[0][0]:.4f}, {layer.coords[0][1]:.4f}, {layer.coords[0][2]:.4f}]")
        print(f"  Last coord:  [{layer.coords[-1][0]:.4f}, {layer.coords[-1][1]:.4f}, {layer.coords[-1][2]:.4f}]")
        print()
    
    # Generate STL to verify geometry
    wing.generate_closed_mesh_stl("test_rotation_offset.stl")
    print("Generated test_rotation_offset.stl")
    
    return wing

if __name__ == "__main__":
    test_rotation_with_offset()
    print("✅ Rotation offset test completed!")
