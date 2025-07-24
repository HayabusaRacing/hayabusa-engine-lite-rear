#!/usr/bin/env python3
"""
Test script to verify the airfoilLayers integration with the GA system
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from geometry.airfoilLayers import airfoilLayers
from geometry.geometryParams import GeometryParams
from config import (AIRFOIL_DENSITY, AIRFOIL_WING_SPAN, AIRFOIL_WING_CHORD, AIRFOIL_FILES,
                   AIRFOIL_SURFACE_DEGREE_U, AIRFOIL_SURFACE_DEGREE_V, AIRFOIL_SAMPLE_RESOLUTION,
                   USE_AIRFOIL_LAYERS)

def test_parameter_length():
    """Test that parameter length calculation works correctly"""
    print("Testing parameter length calculation...")
    
    wing = airfoilLayers(
        density=AIRFOIL_DENSITY, 
        wing_span=AIRFOIL_WING_SPAN, 
        wing_chord=AIRFOIL_WING_CHORD,
        surface_degree_u=AIRFOIL_SURFACE_DEGREE_U,
        surface_degree_v=AIRFOIL_SURFACE_DEGREE_V,
        sample_resolution=AIRFOIL_SAMPLE_RESOLUTION
    )
    
    param_count = wing.get_parameter_number()
    expected_count = AIRFOIL_DENSITY * 5
    
    print(f"Density: {AIRFOIL_DENSITY}")
    print(f"Expected parameters: {expected_count}")
    print(f"Calculated parameters: {param_count}")
    
    assert param_count == expected_count, f"Parameter count mismatch: {param_count} != {expected_count}"
    print("✓ Parameter length calculation works correctly")
    return param_count

def test_geometry_generation():
    """Test that geometry generation works with sample parameters"""
    print("\nTesting geometry generation...")
    
    # Create sample parameters (density=3, so 15 parameters total)
    # Format: [airfoil0, pitch0, x_offset0, z_offset0, scale0, ...]
    sample_params = [
        0, 5.0, 0.0, 0.0, 1.0,   # Layer 1
        0, 10.0, 0.0, 0.0, 0.8,  # Layer 2  
        0, 15.0, 0.0, 0.0, 0.6   # Layer 3
    ]
    
    wing = airfoilLayers(
        density=AIRFOIL_DENSITY, 
        wing_span=AIRFOIL_WING_SPAN, 
        wing_chord=AIRFOIL_WING_CHORD
    )
    
    try:
        output_file = wing.create_geometry_from_array(sample_params, AIRFOIL_FILES, "test_wing.stl")
        print(f"✓ Successfully generated geometry: {output_file}")
        
        # Check if file was created
        if Path(output_file).exists():
            file_size = Path(output_file).stat().st_size
            print(f"✓ STL file created with size: {file_size} bytes")
        else:
            print("✗ STL file was not created")
            
    except Exception as e:
        print(f"✗ Geometry generation failed: {e}")
        raise

def test_parameter_bounds():
    """Test that parameter bounds are reasonable"""
    print("\nTesting parameter bounds...")
    
    wing = airfoilLayers(density=AIRFOIL_DENSITY)
    bounds = wing.get_parameter_bounds()
    
    print("Parameter bounds:")
    for param, (min_val, max_val) in bounds.items():
        print(f"  {param}: [{min_val}, {max_val}]")
    
    print("✓ Parameter bounds retrieved successfully")

def test_config_integration():
    """Test that config values are properly imported"""
    print("\nTesting config integration...")
    
    print(f"USE_AIRFOIL_LAYERS: {USE_AIRFOIL_LAYERS}")
    print(f"AIRFOIL_DENSITY: {AIRFOIL_DENSITY}")
    print(f"AIRFOIL_WING_SPAN: {AIRFOIL_WING_SPAN}")
    print(f"AIRFOIL_WING_CHORD: {AIRFOIL_WING_CHORD}")
    print(f"AIRFOIL_FILES: {AIRFOIL_FILES}")
    
    assert isinstance(USE_AIRFOIL_LAYERS, bool), "USE_AIRFOIL_LAYERS should be boolean"
    assert AIRFOIL_DENSITY > 0, "AIRFOIL_DENSITY should be positive"
    assert len(AIRFOIL_FILES) > 0, "AIRFOIL_FILES should not be empty"
    
    print("✓ Config integration works correctly")

if __name__ == "__main__":
    print("=== Testing airfoilLayers Integration ===")
    
    try:
        test_config_integration()
        param_count = test_parameter_length()
        test_parameter_bounds()
        test_geometry_generation()
        
        print(f"\n✓ All tests passed!")
        print(f"Your GA system is now configured to use airfoilLayers with {param_count} parameters per individual.")
        print(f"You can switch between methods by setting USE_AIRFOIL_LAYERS = {not USE_AIRFOIL_LAYERS} in config.py")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
