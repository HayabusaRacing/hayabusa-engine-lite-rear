"""
Simple script to create matplotlib diagrams of all airfoil .dat files
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from config import AIRFOIL_FILES

def load_airfoil_coordinates(filename):
    """
    Load airfoil coordinates from a .dat file
    
    Args:
        filename: Path to the airfoil .dat file
    
    Returns:
        tuple: (x_coords, y_coords) arrays
    """
    coords = []
    
    # Handle relative paths by checking if file exists, if not try from project root
    filepath = filename
    if not os.path.exists(filepath):
        # Try from project root
        project_root = Path(__file__).resolve().parent
        filepath = project_root / filename
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip() == "" or line.strip().startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                x, y = float(parts[0]), float(parts[1])
                coords.append([x, y])
    except FileNotFoundError:
        print(f"Warning: Could not find file {filepath}")
        return [], []
    
    if not coords:
        return [], []
    
    # Separate x and y coordinates
    x_coords = [pt[0] for pt in coords]
    y_coords = [pt[1] for pt in coords]
    
    return x_coords, y_coords

def plot_single_airfoil(filename, save_path=None, show_points=False):
    """
    Plot a single airfoil and optionally save it
    
    Args:
        filename: Airfoil .dat filename
        save_path: Path to save the plot (optional)
        show_points: Whether to show individual coordinate points
    """
    x_coords, y_coords = load_airfoil_coordinates(filename)
    
    if not x_coords:
        print(f"No data found for {filename}")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot the airfoil shape
    plt.plot(x_coords, y_coords, 'b-', linewidth=2, label='Airfoil Shape')
    
    if show_points:
        plt.scatter(x_coords, y_coords, c='red', s=20, alpha=0.6, label='Data Points')
    
    # Find and mark special points
    leading_edge_idx = x_coords.index(min(x_coords))
    trailing_edge_candidates = [i for i, x in enumerate(x_coords) if abs(x - max(x_coords)) < 1e-6]
    
    # Mark leading edge
    plt.plot(x_coords[leading_edge_idx], y_coords[leading_edge_idx], 
             'go', markersize=10, label='Leading Edge')
    
    # Mark trailing edge(s)
    for idx in trailing_edge_candidates:
        plt.plot(x_coords[idx], y_coords[idx], 'ro', markersize=8, 
                label='Trailing Edge' if idx == trailing_edge_candidates[0] else "")
    
    # Calculate some basic properties
    chord_length = max(x_coords) - min(x_coords)
    max_thickness = max(y_coords) - min(y_coords)
    
    # Formatting
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlabel('Chord Position (x/c)')
    plt.ylabel('Thickness (y/c)')
    plt.title(f'{os.path.basename(filename)}\nChord: {chord_length:.3f}, Max Thickness: {max_thickness:.3f}')
    plt.legend()
    
    # Add some statistics as text
    plt.text(0.02, 0.98, f'Points: {len(x_coords)}\nChord: {chord_length:.4f}\nThickness: {max_thickness:.4f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return plt.gcf()

def create_comparison_plot(airfoil_files, save_path=None):
    """
    Create a comparison plot showing all airfoils together
    
    Args:
        airfoil_files: List of airfoil .dat filenames
        save_path: Path to save the comparison plot
    """
    plt.figure(figsize=(15, 10))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, filename in enumerate(airfoil_files):
        x_coords, y_coords = load_airfoil_coordinates(filename)
        
        if not x_coords:
            continue
        
        color = colors[i % len(colors)]
        label = os.path.basename(filename).replace('.dat', '')
        
        plt.plot(x_coords, y_coords, color=color, linewidth=2, label=label)
        
        # Mark leading edge for each
        leading_edge_idx = x_coords.index(min(x_coords))
        plt.plot(x_coords[leading_edge_idx], y_coords[leading_edge_idx], 
                'o', color=color, markersize=8, alpha=0.7)
    
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlabel('Chord Position (x/c)')
    plt.ylabel('Thickness (y/c)')
    plt.title('Airfoil Comparison - All Available Profiles')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison: {save_path}")
    
    return plt.gcf()

def main():
    """
    Main function to create all airfoil diagrams
    """
    print("=== Creating Airfoil Diagrams ===\n")
    
    # Create output directory
    output_dir = Path("airfoil_diagrams")
    output_dir.mkdir(exist_ok=True)
    
    # Plot individual airfoils
    print("Creating individual airfoil plots...")
    for filename in AIRFOIL_FILES:
        if not filename.endswith('.dat'):
            continue
            
        # Create filename for the plot
        base_name = os.path.basename(filename).replace('.dat', '')
        plot_path = output_dir / f"{base_name}_diagram.png"
        
        # Create and save the plot
        fig = plot_single_airfoil(filename, str(plot_path), show_points=True)
        plt.close(fig)  # Close to free memory
    
    # Create comparison plot
    print("\nCreating comparison plot...")
    comparison_path = output_dir / "airfoil_comparison.png"
    fig = create_comparison_plot(AIRFOIL_FILES, str(comparison_path))
    plt.close(fig)
    
    # Create a detailed analysis plot
    print("\nCreating detailed analysis plot...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detailed Airfoil Analysis', fontsize=16)
    
    for i, filename in enumerate(AIRFOIL_FILES[:4]):  # Limit to first 4 airfoils
        if i >= 4:
            break
            
        x_coords, y_coords = load_airfoil_coordinates(filename)
        if not x_coords:
            continue
        
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        # Plot airfoil
        ax.plot(x_coords, y_coords, 'b-', linewidth=2)
        ax.scatter(x_coords, y_coords, c=range(len(x_coords)), cmap='viridis', s=30, alpha=0.6)
        
        # Mark special points
        leading_edge_idx = x_coords.index(min(x_coords))
        ax.plot(x_coords[leading_edge_idx], y_coords[leading_edge_idx], 'go', markersize=8)
        
        # Find trailing edge points
        max_x = max(x_coords)
        trailing_points = [(j, x, y) for j, (x, y) in enumerate(zip(x_coords, y_coords)) if abs(x - max_x) < 1e-6]
        for j, x, y in trailing_points:
            ax.plot(x, y, 'ro', markersize=6)
        
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_title(f'{os.path.basename(filename)}\n{len(x_coords)} points')
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
        
        # Add point numbering for first few and last few points
        for j in list(range(0, min(5, len(x_coords)))) + list(range(max(5, len(x_coords)-5), len(x_coords))):
            ax.annotate(f'{j}', (x_coords[j], y_coords[j]), xytext=(3, 3), 
                       textcoords='offset points', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    analysis_path = output_dir / "airfoil_detailed_analysis.png"
    plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved detailed analysis: {analysis_path}")
    
    print(f"\n✅ All airfoil diagrams created in '{output_dir}' directory")
    print(f"📁 Files created:")
    for file in sorted(output_dir.glob("*.png")):
        print(f"   - {file.name}")

if __name__ == "__main__":
    main()
