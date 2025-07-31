#!/usr/bin/env python3

"""
Test the new parameter bounds to see if they're reasonable
"""

import sys
sys.path.append('.')

from geometry.airfoilLayers import airfoilLayers
from config import *

def test_bounds():
    print("Testing new parameter bounds...")
    
    wing = airfoilLayers(
        density=AIRFOIL_DENSITY,
        wing_span=AIRFOIL_WING_SPAN,
        wing_chord=AIRFOIL_WING_CHORD,
        y_center=AIRFOIL_Y_CENTER,
        x_center=AIRFOIL_X_CENTER,
        z_center=AIRFOIL_Z_CENTER
    )
    
    bounds = wing.get_parameter_bounds()
    
    print(f"Wing dimensions:")
    print(f"  Wing span: {AIRFOIL_WING_SPAN:.4f} m ({AIRFOIL_WING_SPAN*1000:.1f} mm)")
    print(f"  Wing chord: {AIRFOIL_WING_CHORD:.4f} m ({AIRFOIL_WING_CHORD*1000:.1f} mm)")
    
    print(f"\nParameter bounds:")
    for param, (min_val, max_val) in bounds.items():
        if 'offset' in param:
            print(f"  {param}: ({min_val:.6f}, {max_val:.6f}) m = ({min_val*1000:.2f}, {max_val*1000:.2f}) mm")
        else:
            print(f"  {param}: ({min_val:.3f}, {max_val:.3f})")
    
    print(f"\nOffset ratios:")
    print(f"  Y-offset as % of chord: ±{bounds['y_offset'][1]/AIRFOIL_WING_CHORD*100:.1f}%")
    print(f"  Z-offset as % of span: ±{bounds['z_offset'][1]/AIRFOIL_WING_SPAN*100:.1f}%")

if __name__ == "__main__":
    test_bounds()
