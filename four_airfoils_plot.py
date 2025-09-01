"""
Create a single image with four airfoil diagrams in separate subplots
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

def create_four_airfoil_plot(airfoil_files, save_path="four_airfoils.png"):
    """
    Create a 2x2 subplot image with four airfoils
    
    Args:
        airfoil_files: List of airfoil .dat filenames (uses first 4)
        save_path: Path to save the combined plot
    """
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Airfoil Collection - Four Profiles', fontsize=20, fontweight='bold')
    
    # Colors for each airfoil
    colors = ['navy', 'darkred', 'darkgreen', 'darkorange']
    
    # Take only first 4 airfoils
    airfoils_to_plot = airfoil_files[:4]
    
    for i, filename in enumerate(airfoils_to_plot):
        # Calculate subplot position
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        # Load airfoil data
        x_coords, y_coords = load_airfoil_coordinates(filename)
        
        if not x_coords:
            ax.text(0.5, 0.5, f'No data for\n{filename}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            continue
        
        # Plot the airfoil
        ax.plot(x_coords, y_coords, color=colors[i], linewidth=3, alpha=0.8)
        ax.fill(x_coords, y_coords, color=colors[i], alpha=0.2)
        
        # Find and mark special points
        leading_edge_idx = x_coords.index(min(x_coords))
        trailing_edge_candidates = [j for j, x in enumerate(x_coords) if abs(x - max(x_coords)) < 1e-6]
        
        # Mark leading edge
        ax.plot(x_coords[leading_edge_idx], y_coords[leading_edge_idx], 
               'o', color='lime', markersize=12, markeredgecolor='black', 
               markeredgewidth=2, label='Leading Edge', zorder=10)
        
        # Mark trailing edge(s)
        for idx in trailing_edge_candidates:
            ax.plot(x_coords[idx], y_coords[idx], 's', color='red', 
                   markersize=10, markeredgecolor='black', markeredgewidth=2,
                   label='Trailing Edge' if idx == trailing_edge_candidates[0] else "", zorder=10)
        
        # Calculate properties
        chord_length = max(x_coords) - min(x_coords)
        max_thickness = max(y_coords) - min(y_coords)
        camber = (max(y_coords) + min(y_coords)) / 2
        
        # Formatting
        ax.grid(True, alpha=0.4, linestyle='--')
        ax.set_aspect('equal')
        ax.set_xlabel('Chord Position (x/c)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Thickness (y/c)', fontsize=12, fontweight='bold')
        
        # Clean filename for title
        clean_name = os.path.basename(filename).replace('.dat', '').upper()
        ax.set_title(f'{clean_name}\n{len(x_coords)} points', 
                    fontsize=14, fontweight='bold', pad=15)
        
        # Add statistics box
        stats_text = f'Chord: {chord_length:.3f}\nMax Thick: {max_thickness:.3f}\nCamber: {camber:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                        edgecolor=colors[i], alpha=0.9, linewidth=2))
        
        # Add legend for first subplot only
        if i == 0:
            ax.legend(loc='upper right', fontsize=10)
        
        # Set consistent axis limits for better comparison
        x_margin = 0.05
        y_margin = max_thickness * 0.1
        ax.set_xlim(min(x_coords) - x_margin, max(x_coords) + x_margin)
        ax.set_ylim(min(y_coords) - y_margin, max(y_coords) + y_margin)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for main title
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Four airfoil diagram saved as: {save_path}")
    
    return fig

def create_four_airfoil_detailed(airfoil_files, save_path="four_airfoils_detailed.png"):
    """
    Create a detailed 2x2 subplot with additional analysis
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Detailed Airfoil Analysis - Four Profiles', fontsize=22, fontweight='bold')
    
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']  # Professional colors
    
    airfoils_to_plot = airfoil_files[:4]
    
    for i, filename in enumerate(airfoils_to_plot):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        x_coords, y_coords = load_airfoil_coordinates(filename)
        
        if not x_coords:
            continue
        
        # Main airfoil plot with gradient fill
        ax.plot(x_coords, y_coords, color=colors[i], linewidth=4, alpha=0.9, zorder=5)
        ax.fill(x_coords, y_coords, color=colors[i], alpha=0.3)
        
        # Add point markers with color coding
        n_points = len(x_coords)
        point_colors = plt.cm.viridis(np.linspace(0, 1, n_points))
        ax.scatter(x_coords, y_coords, c=point_colors, s=25, alpha=0.7, zorder=3)
        
        # Special points
        leading_edge_idx = x_coords.index(min(x_coords))
        ax.plot(x_coords[leading_edge_idx], y_coords[leading_edge_idx], 
               'o', color='yellow', markersize=15, markeredgecolor='black', 
               markeredgewidth=3, label='Leading Edge', zorder=10)
        
        # Mark first and last points differently
        ax.plot(x_coords[0], y_coords[0], '^', color='lime', markersize=12, 
               markeredgecolor='black', markeredgewidth=2, label='Start Point', zorder=10)
        ax.plot(x_coords[-1], y_coords[-1], 'v', color='red', markersize=12,
               markeredgecolor='black', markeredgewidth=2, label='End Point', zorder=10)
        
        # Calculate detailed properties
        chord_length = max(x_coords) - min(x_coords)
        max_thickness = max(y_coords) - min(y_coords)
        min_thickness = min(y_coords)
        max_y = max(y_coords)
        
        # Find thickness at various chord positions
        thickness_25 = None
        thickness_50 = None
        thickness_75 = None
        
        for j, x in enumerate(x_coords):
            if abs(x - 0.25) < 0.05 and thickness_25 is None:
                thickness_25 = y_coords[j]
            elif abs(x - 0.50) < 0.05 and thickness_50 is None:
                thickness_50 = y_coords[j]
            elif abs(x - 0.75) < 0.05 and thickness_75 is None:
                thickness_75 = y_coords[j]
        
        # Formatting
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.6, linestyle='--', linewidth=0.3, which='minor')
        ax.minorticks_on()
        ax.set_aspect('equal')
        ax.set_xlabel('Chord Position (x/c)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Thickness (y/c)', fontsize=13, fontweight='bold')
        
        # Enhanced title
        clean_name = os.path.basename(filename).replace('.dat', '').upper()
        ax.set_title(f'{clean_name} Airfoil Profile\n{n_points} coordinate points', 
                    fontsize=15, fontweight='bold', pad=20)
        
        # Detailed statistics box
        stats_text = (f'Chord Length: {chord_length:.4f}\n'
                     f'Max Thickness: {max_thickness:.4f}\n'
                     f'Min Y: {min_thickness:.4f}\n'
                     f'Max Y: {max_y:.4f}')
        
        if thickness_25:
            stats_text += f'\nT @ 25%c: {thickness_25:.4f}'
        if thickness_50:
            stats_text += f'\nT @ 50%c: {thickness_50:.4f}'
        if thickness_75:
            stats_text += f'\nT @ 75%c: {thickness_75:.4f}'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.7', facecolor='white', 
                        edgecolor=colors[i], alpha=0.95, linewidth=2))
        
        # Legend for first subplot
        if i == 0:
            ax.legend(loc='center right', fontsize=10, framealpha=0.9)
        
        # Consistent margins
        x_margin = chord_length * 0.05
        y_margin = max_thickness * 0.15
        ax.set_xlim(min(x_coords) - x_margin, max(x_coords) + x_margin)
        ax.set_ylim(min(y_coords) - y_margin, max(y_coords) + y_margin)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Detailed four airfoil diagram saved as: {save_path}")
    
    return fig

def main():
    """
    Create both versions of the four-airfoil plots
    """
    print("=== Creating Four-Airfoil Diagrams ===\n")
    
    # Create output directory
    output_dir = Path("airfoil_diagrams")
    output_dir.mkdir(exist_ok=True)
    
    # Create standard four-airfoil plot
    standard_path = output_dir / "four_airfoils_combined.png"
    fig1 = create_four_airfoil_plot(AIRFOIL_FILES, str(standard_path))
    plt.close(fig1)
    
    # Create detailed four-airfoil plot
    detailed_path = output_dir / "four_airfoils_detailed.png" 
    fig2 = create_four_airfoil_detailed(AIRFOIL_FILES, str(detailed_path))
    plt.close(fig2)
    
    print(f"\n🎯 Four-airfoil diagrams created:")
    print(f"   📊 Standard: {standard_path}")
    print(f"   🔍 Detailed: {detailed_path}")

if __name__ == "__main__":
    main()
