"""
Test script to verify the end-fixed airfoil layer configuration works correctly.
"""

import sys
import os
from pathlib import Path

# Add the geometry directory to the path
sys.path.append(str(Path(__file__).resolve().parent / "geometry"))

from airfoilLayers import airfoilLayers, geometryParameter
from config import AIRFOIL_FILES, AIRFOIL_END_FIXED

def test_end_fixed_configuration():
    print("=== Testing End-Fixed Airfoil Configuration ===\n")
    
    # Test parameters for density=3 (3 layers: 0, 1, 2)
    # With end-fixed: optimize layers 0 and 1, fix layer 2
    density = 3
    
    # Calculate expected parameter array size
    wing_generator = airfoilLayers(density=density, wing_span=0.025, wing_chord=0.015)
    expected_params = wing_generator.get_parameter_number()
    
    print(f"Density: {density}")
    print(f"Expected parameter count: {expected_params}")
    print(f"This means we optimize {(expected_params // 5)} layers and fix 1 end layer")
    
    # Create a test parameter array
    # Format: [airfoil_idx, pitch, y_offset, z_offset, scale] for each optimizable layer
    param_array = []
    
    # Layer 0 parameters (optimizable)
    param_array.extend([0, 2.0, 0.001, 0.001, 1.0])  # NACA0012, 2° pitch, small offsets
    
    # Layer 1 parameters (optimizable) 
    param_array.extend([1, 4.0, 0.002, 0.002, 0.9])  # E387, 4° pitch, different offsets
    
    # Layer 2 is fixed (not in param_array) - will use AIRFOIL_END_FIXED config
    
    print(f"\nActual parameter array length: {len(param_array)}")
    print(f"Parameter array: {param_array}")
    
    # Test the conversion
    try:
        parameters = wing_generator._array_to_parameters(param_array, AIRFOIL_FILES)
        
        print(f"\nGenerated {len(parameters)} parameter objects:")
        for i, param in enumerate(parameters):
            print(f"  Layer {i}: {os.path.basename(param.wing_type)}, "
                  f"pitch={param.pitch_angle}°, y_offset={param.y_offset:.3f}, "
                  f"z_offset={param.z_offset:.3f}, scale={param.scale:.2f}")
        
        # Verify the end layer uses fixed configuration
        end_param = parameters[-1]
        end_wing_file = os.path.basename(AIRFOIL_FILES[AIRFOIL_END_FIXED['wing_type_idx']])
        expected_end_wing = os.path.basename(end_param.wing_type)
        
        print(f"\nEnd layer verification:")
        print(f"  Expected end wing file: {end_wing_file}")
        print(f"  Actual end wing file: {expected_end_wing}")
        print(f"  Expected pitch: {AIRFOIL_END_FIXED['pitch_angle']}°")
        print(f"  Actual pitch: {end_param.pitch_angle}°")
        
        if (expected_end_wing == end_wing_file and 
            end_param.pitch_angle == AIRFOIL_END_FIXED['pitch_angle'] and
            end_param.y_offset == AIRFOIL_END_FIXED['y_offset'] and
            end_param.z_offset == AIRFOIL_END_FIXED['z_offset'] and
            end_param.scale == AIRFOIL_END_FIXED['scale']):
            print("\n✅ END-FIXED configuration working correctly!")
        else:
            print("\n❌ END-FIXED configuration not working as expected!")
            
        # Test STL generation
        output_file = "test_end_fixed_wing.stl"
        try:
            wing_generator.read_parameters(parameters)
            wing_generator.generate_closed_mesh_stl(output_file)
            print(f"\n✅ STL generation successful: {output_file}")
        except Exception as e:
            print(f"\n❌ STL generation failed: {e}")
            
    except Exception as e:
        print(f"\n❌ Parameter conversion failed: {e}")
        return False
    
    return True

def test_different_densities():
    print("\n=== Testing Different Densities ===\n")
    
    for density in [2, 3, 4, 5]:
        wing_generator = airfoilLayers(density=density, wing_span=0.025, wing_chord=0.015)
        expected_params = wing_generator.get_parameter_number()
        optimizable_layers = expected_params // 5
        
        print(f"Density {density}: {expected_params} params = {optimizable_layers} optimizable + 1 fixed end")

if __name__ == "__main__":
    success = test_end_fixed_configuration()
    test_different_densities()
    
    if success:
        print("\n🎉 All tests passed! End-fixed configuration is working correctly.")
    else:
        print("\n💥 Some tests failed. Please check the implementation.")
