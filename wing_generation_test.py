#!/usr/bin/env python3

"""
Airfoil Generation Test - Thickness Verification

This script tests if the 3mm thickness adjustment in airfoilLayers.py is working correctly.
It generates wings with various airfoil types and scales, then measures the actual thickness
in the output STL files to verify they maintain consistent 3mm thickness.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add the project root to the path so we can import modules
sys.path.append(str(Path(__file__).resolve().parent))

from geometry.airfoilLayers import airfoilLayers
from config import AIRFOIL_FILES, AIRFOIL_WING_SPAN, AIRFOIL_WING_CHORD

try:
    import trimesh
except ImportError:
    print("Error: trimesh library is required. Install with: pip install trimesh")
    sys.exit(1)

def measure_thickness(stl_path):
    """Measure the actual thickness of an airfoil STL file"""
    try:
        mesh = trimesh.load(stl_path)
        
        # Get the bounding box dimensions
        extents = mesh.bounding_box.extents
        
        # Z-dimension is the thickness
        thickness_m = extents[2]
        thickness_mm = thickness_m * 1000
        
        return thickness_mm
    except Exception as e:
        print(f"Error measuring thickness: {e}")
        return None

def test_airfoil_thickness():
    """Test if the airfoil thickness adjustment is working correctly"""
    print("\n=== Testing Airfoil Thickness Adjustment ===\n")
    
    # Create a test directory
    test_dir = Path("test_airfoils")
    os.makedirs(test_dir, exist_ok=True)
    
    # Parameters to test
    airfoil_types = list(range(len(AIRFOIL_FILES)))
    scales = [0.75, 1.0, 1.25]  # Min, middle, max scale values
    
    results = []
    
    # Generate and test each combination
    for airfoil_idx in airfoil_types:
        for scale in scales:
            # Test name and output file
            test_name = f"airfoil_{airfoil_idx}_scale_{scale:.2f}"
            stl_path = test_dir / f"{test_name}.stl"
            
            # Generate parameters for one optimizable layer (not center)
            # [airfoil_idx, pitch_angle, y_offset, z_offset, scale]
            param_array = [airfoil_idx, 0.0, 0.0, 0.0, scale]
            
            print(f"Generating {test_name}...")
            
            # Generate the wing with density=2 (center + one optimizable layer)
            wing = airfoilLayers(
                density=2,  # Center layer + one optimizable layer
                wing_span=AIRFOIL_WING_SPAN,
                wing_chord=AIRFOIL_WING_CHORD
            )
            wing.create_geometry_from_array(param_array, AIRFOIL_FILES, str(stl_path))

            wing.visualize_airfoils("debug_airfoils.png")
            
            # Measure the thickness
            thickness = measure_thickness(stl_path)
            
            if thickness is not None:
                results.append({
                    'airfoil': AIRFOIL_FILES[airfoil_idx],
                    'scale': scale,
                    'thickness_mm': thickness,
                    'target_thickness': 3.0,
                    'difference_mm': thickness - 3.0,
                    'stl_path': stl_path
                })
                
                print(f"  Airfoil: {AIRFOIL_FILES[airfoil_idx]}")
                print(f"  Scale: {scale}")
                print(f"  Measured thickness: {thickness:.2f}mm")
                print(f"  Difference from target: {thickness - 3.0:.2f}mm")
                print("")
    
    # Print summary
    print("\n=== Thickness Test Results ===\n")
    print(f"{'Airfoil':<20} {'Scale':<6} {'Thickness':<10} {'Diff from 3mm':<15}")
    print("-" * 55)
    
    for result in results:
        print(f"{os.path.basename(result['airfoil']):<20} {result['scale']:<6.2f} "
              f"{result['thickness_mm']:<10.2f}mm {result['difference_mm']:<+15.2f}mm")
    
    # Calculate statistics
    thicknesses = [result['thickness_mm'] for result in results]
    differences = [result['difference_mm'] for result in results]
    
    print("\n=== Statistics ===\n")
    print(f"Average thickness: {np.mean(thicknesses):.2f}mm")
    print(f"Standard deviation: {np.std(thicknesses):.2f}mm")
    print(f"Min thickness: {min(thicknesses):.2f}mm")
    print(f"Max thickness: {max(thicknesses):.2f}mm")
    print(f"Average difference from target: {np.mean(differences):.2f}mm")
    
    # Overall test result
    avg_abs_diff = np.mean([abs(diff) for diff in differences])
    if avg_abs_diff < 0.1:  # Within 0.1mm tolerance
        print("\n✅ TEST PASSED: Thickness adjustment is working correctly!")
    else:
        print("\n❌ TEST FAILED: Thickness adjustment is not consistently achieving 3mm!")
        print(f"   Average absolute difference: {avg_abs_diff:.2f}mm")
        
    return results

if __name__ == "__main__":
    test_airfoil_thickness()