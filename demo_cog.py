#!/usr/bin/env python3

"""
Example usage of calculate_cog.py script
"""

import subprocess
import json
import os

def demo_cog_calculation():
    """
    Demonstrate center of gravity calculation
    """
    print("=== Center of Gravity Calculation Demo ===\n")
    
    # Example 1: Basic surface centroid calculation
    print("1. Basic surface centroid calculation:")
    result = subprocess.run([
        "python", "utils/calculate_cog.py", 
        "--file", "utils/wing.stl"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"Error: {result.stderr}")
    
    # Example 2: Volumetric calculation with JSON output
    print("\n2. Volumetric calculation with JSON output:")
    result = subprocess.run([
        "python", "utils/calculate_cog.py", 
        "--file", "utils/wing.stl",
        "--volumetric",
        "--output", "example_cog.json"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(result.stdout)
        
        # Read and display JSON results
        if os.path.exists("example_cog.json"):
            with open("example_cog.json", "r") as f:
                data = json.load(f)
            
            print("\nJSON Output Summary:")
            print(f"Surface Centroid: ({data['surface_centroid']['x']:.6f}, "
                  f"{data['surface_centroid']['y']:.6f}, "
                  f"{data['surface_centroid']['z']:.6f})")
            print(f"Surface Area: {data['surface_area']:.6f}")
            
            if 'volumetric_centroid' in data:
                print(f"Volumetric Centroid: ({data['volumetric_centroid']['x']:.6f}, "
                      f"{data['volumetric_centroid']['y']:.6f}, "
                      f"{data['volumetric_centroid']['z']:.6f})")
            
            # Clean up
            os.remove("example_cog.json")
    else:
        print(f"Error: {result.stderr}")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    demo_cog_calculation()
