import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def fix_airfoil_format(filename, output_dir=None):
    """
    Fix airfoil DAT file format to ensure standard ordering:
    1. Start at trailing edge (x=1.0)
    2. Go counter-clockwise along upper surface to leading edge (x~0)
    3. Continue from leading edge along lower surface back to trailing edge
    """
    # Read the file
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Extract coordinates, skipping comments
    coords = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('//'):
            continue
        try:
            parts = line.split()
            if len(parts) >= 2:
                x, y = float(parts[0]), float(parts[1])
                coords.append((x, y))
        except ValueError:
            continue
    
    if not coords:
        print(f"No valid coordinates found in {filename}")
        return None
    
    # Check format by looking at first and last x-coordinates
    first_x = coords[0][0]
    last_x = coords[-1][0]
    
    needs_reordering = False
    
    # Standard format should start and end at x=1.0 (trailing edge)
    # If it starts at x~0 (leading edge), it needs reordering
    if abs(first_x) < 0.1 and abs(last_x - 1.0) < 0.1:
        print(f"{filename} has inverted format - fixing...")
        needs_reordering = True
    
    if needs_reordering:
        # Find the point closest to x=1.0 (trailing edge)
        x_values = [pt[0] for pt in coords]
        te_idx = x_values.index(max(x_values))
        
        # Reorder points: [te_idx:] + [0:te_idx]
        # This takes points from TE to LE and adds points from LE to TE
        new_coords = coords[te_idx:] + coords[:te_idx]
        
        # Ensure we really start and end at trailing edge
        if abs(new_coords[0][0] - 1.0) > 0.01 or abs(new_coords[-1][0] - 1.0) > 0.01:
            print(f"Warning: Unable to properly identify trailing edge in {filename}")
            print(f"First point: {new_coords[0]}, Last point: {new_coords[-1]}")
    else:
        new_coords = coords
        print(f"{filename} already has standard format.")
    
    # Generate output filename
    basename = os.path.basename(filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, basename)
    else:
        name_parts = os.path.splitext(basename)
        output_file = os.path.join(os.path.dirname(filename), 
                                  f"{name_parts[0]}_fixed{name_parts[1]}")
    
    # Write the corrected file
    with open(output_file, 'w') as f:
        for x, y in new_coords:
            f.write(f"{x:.8f}  {y:.8f}\n")
    
    print(f"Fixed airfoil saved to {output_file}")
    
    # Visualize the airfoil to confirm
    plot_airfoil(coords, new_coords, basename, output_file)
    
    return output_file

def plot_airfoil(original, fixed, title, output_path):
    """Plot original and fixed airfoil for comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original airfoil
    x_orig = [pt[0] for pt in original]
    y_orig = [pt[1] for pt in original]
    ax1.plot(x_orig, y_orig, 'b-', label='Profile')
    ax1.plot(x_orig, y_orig, 'r.', markersize=3, label='Points')
    ax1.plot(x_orig[0], y_orig[0], 'go', markersize=6, label='Start')
    ax1.plot(x_orig[-1], y_orig[-1], 'mo', markersize=6, label='End')
    ax1.set_title(f"Original: {title}")
    ax1.grid(True)
    ax1.axis('equal')
    ax1.legend()
    
    # Fixed airfoil
    x_fixed = [pt[0] for pt in fixed]
    y_fixed = [pt[1] for pt in fixed]
    ax2.plot(x_fixed, y_fixed, 'b-', label='Profile')
    ax2.plot(x_fixed, y_fixed, 'r.', markersize=3, label='Points')
    ax2.plot(x_fixed[0], y_fixed[0], 'go', markersize=6, label='Start')
    ax2.plot(x_fixed[-1], y_fixed[-1], 'mo', markersize=6, label='End')
    ax2.set_title(f"Fixed: {title}")
    ax2.grid(True)
    ax2.axis('equal')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save the comparison plot
    plot_path = os.path.splitext(output_path)[0] + "_comparison.png"
    plt.savefig(plot_path)
    print(f"Comparison plot saved to {plot_path}")
    plt.close()

def main():
    # Path to your airfoil files
    airfoil_dir = Path(__file__).resolve().parent
    
    # Create output directory
    output_dir = airfoil_dir / "fixed_airfoils"
    os.makedirs(output_dir, exist_ok=True)
    
    # Specific airfoil to fix
    lorentz_file = airfoil_dir / "lorentz_teardrop.dat"
    if os.path.exists(lorentz_file):
        fix_airfoil_format(str(lorentz_file), str(output_dir))
    else:
        print(f"Could not find {lorentz_file}")
    
    # Optionally check all airfoil files
    check_all = input("Check all DAT files in directory? (y/n): ").strip().lower()
    if check_all == 'y':
        for file in airfoil_dir.glob("*.dat"):
            if file.name != "lorentz_teardrop.dat":  # Skip already processed
                print(f"\nChecking {file.name}...")
                fix_airfoil_format(str(file), str(output_dir))

if __name__ == "__main__":
    main()