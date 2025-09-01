"""
Simple four-airfoil plot - clean and minimal
"""

import matplotlib.pyplot as plt
import os
from pathlib import Path
from config import AIRFOIL_FILES

def load_airfoil_simple(filename):
    """Load airfoil coordinates"""
    x_coords, y_coords = [], []
    
    filepath = filename
    if not os.path.exists(filepath):
        project_root = Path(__file__).resolve().parent
        filepath = project_root / filename
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    x_coords.append(float(parts[0]))
                    y_coords.append(float(parts[1]))
    
    return x_coords, y_coords

def create_simple_four_plot():
    """Create a clean 2x2 airfoil plot"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Airfoil Profiles Collection', fontsize=18, fontweight='bold')
    
    for i, filename in enumerate(AIRFOIL_FILES[:4]):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        x_coords, y_coords = load_airfoil_simple(filename)
        
        if x_coords:
            # Simple clean plot
            ax.plot(x_coords, y_coords, 'b-', linewidth=2.5)
            ax.fill(x_coords, y_coords, alpha=0.2, color='lightblue')
            
            # Mark leading and trailing edges
            le_idx = x_coords.index(min(x_coords))
            ax.plot(x_coords[le_idx], y_coords[le_idx], 'go', markersize=8, label='LE')
            ax.plot(x_coords[0], y_coords[0], 'ro', markersize=6, label='TE')
            ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=6)
        
        # Clean formatting
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
        
        # Simple title
        name = os.path.basename(filename).replace('.dat', '').upper()
        ax.set_title(name, fontsize=14, fontweight='bold')
        
        if i == 0:  # Legend only on first subplot
            ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save
    output_path = "airfoil_diagrams/four_airfoils_simple.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✅ Simple four-airfoil plot saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    create_simple_four_plot()
