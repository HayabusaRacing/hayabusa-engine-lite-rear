"""
Simple airfoil plotter - minimal version for quick visualization
"""

import matplotlib.pyplot as plt
import os
from pathlib import Path

def plot_airfoil_simple(filename, save_as_png=True):
    """
    Simple function to plot an airfoil from a .dat file
    
    Args:
        filename: Name of the .dat file (e.g., "naca0012.dat")
        save_as_png: Whether to save as PNG file
    """
    # Load coordinates
    x_coords, y_coords = [], []
    
    # Try to find the file
    filepath = filename
    if not os.path.exists(filepath):
        project_root = Path(__file__).resolve().parent
        filepath = project_root / filename
    
    # Read the file
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    x_coords.append(float(parts[0]))
                    y_coords.append(float(parts[1]))
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords, y_coords, 'b-', linewidth=2)
    plt.scatter(x_coords[0], y_coords[0], color='red', s=100, label='Start Point', zorder=5)
    plt.scatter(x_coords[-1], y_coords[-1], color='green', s=100, label='End Point', zorder=5)
    
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlabel('x/c')
    plt.ylabel('y/c')
    plt.title(f'Airfoil: {os.path.basename(filename)}')
    plt.legend()
    
    if save_as_png:
        output_name = f"{os.path.basename(filename).replace('.dat', '')}_simple.png"
        plt.savefig(output_name, dpi=200, bbox_inches='tight')
        print(f"Saved: {output_name}")
    
    plt.show()

# Quick usage examples:
if __name__ == "__main__":
    # Plot all airfoils from config
    from config import AIRFOIL_FILES
    
    for airfoil in AIRFOIL_FILES:
        try:
            plot_airfoil_simple(airfoil, save_as_png=True)
            plt.close()  # Close the figure to save memory
        except Exception as e:
            print(f"Error plotting {airfoil}: {e}")
    
    print("✅ Simple airfoil plots completed!")
