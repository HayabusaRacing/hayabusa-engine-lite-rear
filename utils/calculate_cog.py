#!/usr/bin/env python3

"""
Calculate Center of Gravity (Centroid) of STL Files

This script calculates the center of gravity (centroid) of an STL file.
For hollow objects, it calculates the centroid of the surface mesh.
For solid objects, it can calculate the volumetric centroid.

Usage:
    python calculate_cog.py --file path/to/file.stl
    python calculate_cog.py --file path/to/file.stl --volumetric
    python calculate_cog.py --file path/to/file.stl --output results.json
"""

import argparse
import json
import numpy as np
import sys
import os

try:
    import trimesh
except ImportError:
    print("Error: trimesh library is required. Install with: pip install trimesh")
    sys.exit(1)

def calculate_surface_centroid(mesh):
    """
    Calculate the centroid of the surface mesh (area-weighted center of triangular faces)
    """
    if len(mesh.faces) == 0:
        raise ValueError("Mesh has no faces")
    
    # Get face centers and areas
    face_centers = mesh.triangles_center
    face_areas = mesh.area_faces
    
    # Calculate area-weighted centroid
    total_area = np.sum(face_areas)
    if total_area == 0:
        raise ValueError("Total surface area is zero")
    
    weighted_centers = face_centers * face_areas.reshape(-1, 1)
    surface_centroid = np.sum(weighted_centers, axis=0) / total_area
    
    return surface_centroid, total_area

def calculate_volumetric_centroid(mesh):
    """
    Calculate the volumetric center of mass (requires watertight mesh)
    """
    if not mesh.is_watertight:
        print("Warning: Mesh is not watertight. Volumetric centroid may be inaccurate.")
    
    if mesh.volume == 0:
        raise ValueError("Mesh volume is zero")
    
    return mesh.center_mass, mesh.volume

def load_mesh(file_path):
    """
    Load STL file and return trimesh object
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not file_path.lower().endswith('.stl'):
        raise ValueError("File must be an STL file")
    
    try:
        mesh = trimesh.load_mesh(file_path)
        return mesh
    except Exception as e:
        raise RuntimeError(f"Failed to load STL file: {e}")

def format_results(surface_centroid, surface_area, volumetric_centroid=None, volume=None, mesh_info=None):
    """
    Format results for output
    """
    results = {
        "surface_centroid": {
            "x": float(surface_centroid[0]),
            "y": float(surface_centroid[1]), 
            "z": float(surface_centroid[2])
        },
        "surface_area": float(surface_area),
        "mesh_info": mesh_info or {}
    }
    
    if volumetric_centroid is not None:
        results["volumetric_centroid"] = {
            "x": float(volumetric_centroid[0]),
            "y": float(volumetric_centroid[1]),
            "z": float(volumetric_centroid[2])
        }
        results["volume"] = float(volume) if volume is not None else None
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Calculate center of gravity of STL files")
    parser.add_argument("--file", "-f", type=str, required=True, 
                       help="Path to STL file")
    parser.add_argument("--volumetric", "-v", action="store_true",
                       help="Calculate volumetric centroid (requires watertight mesh)")
    parser.add_argument("--output", "-o", type=str,
                       help="Output results to JSON file")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed mesh information")
    
    args = parser.parse_args()
    
    try:
        # Load mesh
        print(f"Loading STL file: {args.file}")
        mesh = load_mesh(args.file)
        
        # Get mesh info
        mesh_info = {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "is_watertight": mesh.is_watertight,
            "bounds": {
                "min": mesh.bounds[0].tolist(),
                "max": mesh.bounds[1].tolist()
            }
        }
        
        # Add is_valid if available
        if hasattr(mesh, 'is_valid'):
            mesh_info["is_valid"] = mesh.is_valid
        
        if args.verbose:
            print(f"Mesh info:")
            print(f"  Vertices: {mesh_info['vertices']}")
            print(f"  Faces: {mesh_info['faces']}")
            print(f"  Is watertight: {mesh_info['is_watertight']}")
            if 'is_valid' in mesh_info:
                print(f"  Is valid: {mesh_info['is_valid']}")
            print(f"  Bounds: {mesh_info['bounds']}")
        
        # Calculate surface centroid
        print("Calculating surface centroid...")
        surface_centroid, surface_area = calculate_surface_centroid(mesh)
        
        volumetric_centroid = None
        volume = None
        
        # Calculate volumetric centroid if requested
        if args.volumetric:
            print("Calculating volumetric centroid...")
            try:
                volumetric_centroid, volume = calculate_volumetric_centroid(mesh)
            except Exception as e:
                print(f"Warning: Could not calculate volumetric centroid: {e}")
        
        # Format results
        results = format_results(surface_centroid, surface_area, 
                               volumetric_centroid, volume, mesh_info)
        
        # Print results
        print("\n" + "="*50)
        print("CENTER OF GRAVITY RESULTS")
        print("="*50)
        print(f"Surface Centroid (Area-weighted):")
        print(f"  X: {surface_centroid[0]:.6f}")
        print(f"  Y: {surface_centroid[1]:.6f}")
        print(f"  Z: {surface_centroid[2]:.6f}")
        print(f"Surface Area: {surface_area:.6f}")
        
        if volumetric_centroid is not None:
            print(f"\nVolumetric Centroid (Mass center):")
            print(f"  X: {volumetric_centroid[0]:.6f}")
            print(f"  Y: {volumetric_centroid[1]:.6f}")
            print(f"  Z: {volumetric_centroid[2]:.6f}")
            if volume is not None:
                print(f"Volume: {volume:.6f}")
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")
        
        return results
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
