from geomdl import BSpline
from geomdl import utilities
from geomdl import exchange
from geomdl.visualization import VisMPL
import numpy as np
import struct
import math

# Import config for fixed center airfoil parameters
try:
    from config import AIRFOIL_CENTER_FIXED
except ImportError:
    # Fallback if config not available
    AIRFOIL_CENTER_FIXED = {
        'wing_type_idx': 0,
        'pitch_angle': 0.0,
        'y_offset': 0.0,
        'z_offset': 0.0,
        'scale': 1.0
    }

class geometryParameter:
    def __init__(self, wing_type, pitch_angle=0.0, y_offset=0.0, z_offset=0.0, scale=1.0):
        self.wing_type = wing_type
        self.pitch_angle = pitch_angle
        self.y_offset = y_offset  # Now adjustable per layer (was x_offset)
        self.z_offset = z_offset
        self.scale = scale


class airfoilLayer:
    def __init__(self, filename, y_offset=0.0, x_offset=0.0, z_offset=0.0, scale=1.0):
        self.filename = filename
        self.y_offset = y_offset  # Now adjustable per layer (was x_offset)
        self.x_offset = x_offset  # Now span direction (was y_offset)
        self.z_offset = z_offset
        self.scale = scale
        self.coords = self.load_coords()

    def load_coords(self):
        coords = []
        # Handle relative paths by checking if file exists, if not try from project root
        import os
        filepath = self.filename
        if not os.path.exists(filepath):
            # Try from project root
            from pathlib import Path
            project_root = Path(__file__).resolve().parent.parent
            filepath = project_root / self.filename

        with open(filepath, 'r') as f:
            for line in f:
                if line.strip() == "" or line.strip().startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                x, z = float(parts[0]), float(parts[1])
                coords.append([self.x_offset, x * self.scale + self.y_offset, z * self.scale + self.z_offset])

        # Resample for uniform point distribution
        coords = airfoilLayer.resample_points(coords, num_points=100)

        # Adjust thickness to fixed 3mm (0.003 units)
        target_thickness = 0.003

        # Get current z range (thickness)
        z_values = [pt[2] for pt in coords]
        min_z = min(z_values)
        max_z = max(z_values)
        current_thickness = max_z - min_z

        # Calculate scaling factor
        if current_thickness > 1e-6:  # Avoid division by zero
            # Find z-center of the airfoil
            z_center = (max_z + min_z) / 2
            scale_factor = target_thickness / current_thickness

            # Scale z-coordinates around the center
            for pt in coords:
                # Shift to center, scale, shift back
                pt[2] = (pt[2] - z_center) * scale_factor + z_center

        return coords
    
    def rotate_pitch(self, angle, y_center=None, z_center=None):
        # Rotate around the layer's offset position (default) or specified center
        if y_center is None:
            y_center = self.y_offset
        if z_center is None:
            z_center = self.z_offset
            
        angle_rad = math.radians(angle)
        cos_angle, sin_angle = math.cos(angle_rad), math.sin(angle_rad)
        
        for pt in self.coords:
            x, y, z = pt
            # Translate to rotation center, rotate, then translate back
            y_rel = y - y_center
            z_rel = z - z_center
            
            y_rotated = y_rel * cos_angle - z_rel * sin_angle
            z_rotated = y_rel * sin_angle + z_rel * cos_angle
            
            pt[1] = y_rotated + y_center  # Y becomes the chord direction
            pt[2] = z_rotated + z_center
        return self.coords
    
    def apply_transformation(self, transformation_matrix):
        for i, pt in enumerate(self.coords):
            pt_homogeneous = np.array([pt[0], pt[1], pt[2], 1.0])
            transformed = transformation_matrix @ pt_homogeneous
            self.coords[i] = transformed[:3].tolist()
        return self.coords
    
    @staticmethod
    def resample_points(coords, num_points=100):
        """Resample airfoil with properly closed trailing edge"""
        import numpy as np

        if len(coords) < 3:
            return coords

        # Find trailing edge - usually the point with max X (or maybe both first and last points)
        x_values = [p[1] for p in coords]  # Y-coordinate holds the chord direction
        max_x_idx = x_values.index(max(x_values))

        # Reorder points to start and end at trailing edge if needed
        if max_x_idx != 0 and max_x_idx != len(coords)-1:
            coords = coords[max_x_idx:] + coords[:max_x_idx]

        # Find leading edge (minimum Y)
        x_values = [p[1] for p in coords]
        min_x_idx = x_values.index(min(x_values))

        # Calculate distances as before...
        distances = [0.0]
        for i in range(1, len(coords)):
            p1 = np.array(coords[i-1])
            p2 = np.array(coords[i])
            distances.append(distances[-1] + np.linalg.norm(p2 - p1))

        total_length = distances[-1]
        if total_length == 0:
            return coords

        # Create new points, but reserve first and last for exact trailing edge
        new_coords = []

        # First point is trailing edge
        new_coords.append(list(coords[0]))

        # Middle points - use num_points-2 since we're adding TE points manually
        for i in range(1, num_points-1):
            target_dist = total_length * i / (num_points - 1)

            # Find the segment
            idx = 0
            while idx < len(distances)-1 and distances[idx+1] < target_dist:
                idx += 1

            if idx >= len(distances)-1:
                continue

            segment_length = distances[idx+1] - distances[idx]
            if segment_length > 0:
                t = (target_dist - distances[idx]) / segment_length
            else:
                t = 0

            p1 = np.array(coords[idx])
            p2 = np.array(coords[idx+1])
            new_point = p1 + t * (p2 - p1)
            new_coords.append(new_point.tolist())

        # Last point MUST match first for closed trailing edge
        new_coords.append(list(new_coords[0]))

        return new_coords
    
class airfoilLayers:
    def __init__(self, density, y_center=0.0, x_center=0.0, z_center=0.0, wing_span=0.0, wing_chord=0.0, 
                 surface_degree_u=3, surface_degree_v=2, sample_resolution=40):
        self.y_center = y_center  # Center in adjustable direction (was x_center)
        self.x_center = x_center  # Center in span direction (was y_center)
        self.z_center = z_center
        self.wing_span = wing_span
        self.wing_chord = wing_chord
        self.density = density
        self.surface_degree_u = surface_degree_u
        self.surface_degree_v = surface_degree_v
        self.sample_resolution = sample_resolution
        self.layers = []

    def add_layer(self, layer: airfoilLayer, index=-1):
        if not isinstance(layer, airfoilLayer):
            raise TypeError("Layer must be an instance of airfoilLayer")
        
        if index == -1:
            self.layers.append(layer)
        elif 0 <= index <= len(self.layers):
            self.layers.insert(index, layer)
        else:
            raise IndexError(f"Index {index} out of range for {len(self.layers)} layers")
    
    def read_parameters(self, parameters: list[geometryParameter]):
        self._validate_parameters(parameters)
        self.layers.clear()
        
        for i, param in enumerate(parameters):
            x_offset = self.x_center + (self.wing_span / (self.density - 1)) * i  # Fixed span spacing
            layer = self._create_layer(param, x_offset)
            layer.rotate_pitch(param.pitch_angle)
            self.add_layer(layer)
            
            if i != 0:
                negative_x_offset = self.x_center - (self.wing_span / (self.density - 1)) * i
                symmetric_layer = self._create_layer(param, negative_x_offset)
                symmetric_layer.rotate_pitch(param.pitch_angle)
                self.add_layer(symmetric_layer, 0)
    
    def _validate_parameters(self, parameters):
        if not all(isinstance(param, geometryParameter) for param in parameters):
            raise TypeError("All items must be geometryParameter instances")
        if len(parameters) != self.density:
            raise ValueError(f"Expected {self.density} parameters, got {len(parameters)}")
    
    def _create_layer(self, param, x_offset):
        return airfoilLayer(param.wing_type, 
                          y_offset=param.y_offset + self.y_center,  # Add class-level y_center
                          x_offset=x_offset, 
                          z_offset=param.z_offset + self.z_center,  # Add class-level z_center
                          scale=self.wing_chord * param.scale)
    
    def _create_surface(self):
        print("\n[DEBUG] Layer point counts before surface creation:")
        for i, layer in enumerate(self.layers):
            import os
            print(f"Layer {i} ({os.path.basename(layer.filename)}): {len(layer.coords)} points")
        sections = [layer.coords for layer in self.layers]
        ctrlpts2d = list(map(list, zip(*sections)))
        ctrlpts_flat = [pt for row in ctrlpts2d for pt in row]

        surf = BSpline.Surface()
        surf.degree_u = self.surface_degree_u
        surf.degree_v = min(self.surface_degree_v, len(self.layers) - 1)

        nu = len(ctrlpts2d)
        nv = len(ctrlpts2d[0])
        surf.set_ctrlpts(ctrlpts_flat, nu, nv)

        surf.knotvector_u = utilities.generate_knot_vector(surf.degree_u, nu)
        surf.knotvector_v = utilities.generate_knot_vector(surf.degree_v, nv)
        surf.sample_size = self.sample_resolution

        return surf
    
    def generate_stl(self, filename="airfoil_layers.stl"):
        surf = self._create_surface()
        exchange.export_stl(surf, filename)
        print(f"STL file exported successfully as '{filename}'")

    def generate_closed_mesh_stl(self, filename="airfoil_closed_mesh.stl"):
        surf = self._create_surface()
        self._write_closed_mesh_stl(surf, filename)
        print(f"Closed mesh STL file exported successfully as '{filename}'")
    
    def get_parameter_number(self):
        # Only optimize outer layers (indices 1 to density-1), center (index 0) is fixed
        return (self.density - 1) * 5

    def _write_closed_mesh_stl(self, surf, filename):
        surf.evaluate()
        points = np.array(surf.evalpts)
        
        u_count, v_count = self._get_surface_dimensions(surf, points)
        point_grid = points.reshape(v_count, u_count, 3)
        
        triangles = self._generate_surface_triangles(point_grid, u_count, v_count)
        triangles.extend(self._generate_end_caps())
        
        self._write_stl_file(triangles, filename)

    def _get_surface_dimensions(self, surf, points):
        if hasattr(surf, 'sample_size_u') and hasattr(surf, 'sample_size_v'):
            return surf.sample_size_u, surf.sample_size_v
        else:
            total_points = len(points)
            u_count = int(np.sqrt(total_points))
            v_count = total_points // u_count
            return u_count, v_count

    def _generate_surface_triangles(self, point_grid, u_count, v_count):
        triangles = []
        for i in range(v_count - 1):
            for j in range(u_count - 1):
                p1, p2, p3, p4 = point_grid[i, j], point_grid[i + 1, j], point_grid[i, j + 1], point_grid[i + 1, j + 1]
                triangles.extend([[p1, p2, p3], [p2, p4, p3]])
        return triangles

    def _generate_end_caps(self):
        triangles = []
        for section, reverse in [(self.layers[0].coords, True), (self.layers[-1].coords, False)]:
            if len(section) >= 3:
                center = np.mean(section, axis=0)
                for i in range(len(section) - 1):
                    if reverse:
                        triangles.append([center, section[i + 1], section[i]])
                    else:
                        triangles.append([center, section[i], section[i + 1]])
        return triangles

    def _write_stl_file(self, triangles, filename):
        with open(filename, 'wb') as f:
            f.write(b'\x00' * 80)
            f.write(struct.pack('<I', len(triangles)))
            
            for triangle in triangles:
                normal = self._calculate_normal(triangle)
                f.write(struct.pack('<fff', *normal))
                for vertex in triangle:
                    f.write(struct.pack('<fff', *vertex))
                f.write(b'\x00\x00')

    def _calculate_normal(self, triangle):
        v1 = np.array(triangle[1]) - np.array(triangle[0])
        v2 = np.array(triangle[2]) - np.array(triangle[0])
        normal = np.cross(v1, v2)
        norm_length = np.linalg.norm(normal)
        return normal / norm_length if norm_length > 0 else [0, 0, 0]

    def visualize(self):
        surf = self._create_surface()
        surf.delta = 0.02
        surf.vis = VisMPL.VisSurface()
        surf.render()

    def get_layer_count(self):
        return len(self.layers)

    def get_surface_info(self):
        return {
            'layer_count': len(self.layers),
            'surface_degree_u': self.surface_degree_u,
            'surface_degree_v': min(self.surface_degree_v, len(self.layers) - 1),
            'sample_resolution': self.sample_resolution,
            'wing_span': self.wing_span,
            'wing_chord': self.wing_chord
        }

    def create_geometry_from_array(self, param_array, airfoil_files, output_filename="wing.stl"):
        parameters = self._array_to_parameters(param_array, airfoil_files)
        self.read_parameters(parameters)
        self.generate_closed_mesh_stl(output_filename)
        return output_filename
    
    def _array_to_parameters(self, param_array, airfoil_files):
        parameters = []
        param_per_layer = 5
        expected_params = (self.density - 1) * param_per_layer  # Only outer layers are optimized
        
        if len(param_array) != expected_params:
            raise ValueError(f"Expected {expected_params} parameters for {self.density-1} optimizable layers, got {len(param_array)}")
        
        # Add fixed center airfoil (index 0)
        center_wing_type = airfoil_files[AIRFOIL_CENTER_FIXED['wing_type_idx'] % len(airfoil_files)]
        center_param = geometryParameter(
            center_wing_type,
            AIRFOIL_CENTER_FIXED['pitch_angle'],
            AIRFOIL_CENTER_FIXED['y_offset'],
            AIRFOIL_CENTER_FIXED['z_offset'],
            AIRFOIL_CENTER_FIXED['scale']
        )
        parameters.append(center_param)
        
        # Add optimizable outer layers (indices 1 to density-1)
        for i in range(1, self.density):
            # Array index for this layer (0-based for param_array since center is not in array)
            array_layer_idx = i - 1
            base_idx = array_layer_idx * param_per_layer
            
            wing_type_idx = int(param_array[base_idx])
            wing_type = airfoil_files[wing_type_idx % len(airfoil_files)]
            
            pitch_angle = param_array[base_idx + 1]
            y_offset = param_array[base_idx + 2]  # Adjustable per layer
            z_offset = param_array[base_idx + 3]
            scale = param_array[base_idx + 4]
            
            parameters.append(geometryParameter(wing_type, pitch_angle, y_offset, z_offset, scale))
        
        return parameters
    
    def get_parameter_bounds(self):
        """
        Dynamically retrieve parameter bounds using configuration values from config.py.
        """
        from config import (
            AIRFOIL_FILES,
            PITCH_ANGLE_BOUNDS,
            Y_OFFSET_BOUNDS,
            Z_OFFSET_BOUNDS,
            SCALE_BOUNDS
        )

        # Dynamically calculate the airfoil type range based on the number of airfoil files
        airfoil_type_bounds = (0, len(AIRFOIL_FILES) - 1)

        return {
            'wing_type_idx': airfoil_type_bounds,  # Dynamically set based on airfoil files
            'pitch_angle': PITCH_ANGLE_BOUNDS,    # From config.py
            'y_offset': Y_OFFSET_BOUNDS,          # From config.py
            'z_offset': Z_OFFSET_BOUNDS,          # From config.py
            'scale': SCALE_BOUNDS                 # From config.py
        }
    
    def get_array_size(self):
        return self.density * 5

    @staticmethod
    def quick_generate(param_array, airfoil_files, density=3, wing_span=1.0, wing_chord=0.5, output_filename="wing.stl"):
        wing = airfoilLayers(density=density, wing_span=wing_span, wing_chord=wing_chord)
        return wing.create_geometry_from_array(param_array, airfoil_files, output_filename)
    
    def visualize_airfoils(self, output_file=None):
        """Generate detailed visualizations of all airfoil layers for debugging"""
        import matplotlib.pyplot as plt
        from matplotlib.path import Path
        import matplotlib.patches as patches
        import os
        import numpy as np

        n_layers = len(self.layers)
        if n_layers == 0:
            print("No layers to visualize")
            return

        # Create figure with subplots - one row per layer
        fig, axes = plt.subplots(n_layers, 2, figsize=(15, 4*n_layers))
        if n_layers == 1:
            axes = [axes]  # Make it 2D for consistent indexing

        # Plot each layer
        for i, layer in enumerate(self.layers):
            coords = layer.coords
            if not coords:
                continue

            # Get layer name from filename
            layer_name = os.path.basename(layer.filename)

            # Extract coordinates for easier plotting
            x_span = [p[0] for p in coords]
            y_chord = [p[1] for p in coords]
            z_height = [p[2] for p in coords]

            # Find special points
            min_y_idx = y_chord.index(min(y_chord))  # Leading edge
            max_y_idx = y_chord.index(max(y_chord))  # Trailing edge (possibly)

            # 1. SIDE VIEW - airfoil profile (Y-Z plane)
            ax1 = axes[i][0]

            # Plot the airfoil profile
            ax1.plot(y_chord, z_height, 'b-', linewidth=1, alpha=0.7)
            ax1.scatter(y_chord, z_height, c=range(len(coords)), cmap='viridis', 
                       s=50, zorder=5, alpha=0.7)

            # Mark first and last points
            ax1.plot(y_chord[0], z_height[0], 'go', markersize=10, label='First Point')
            ax1.plot(y_chord[-1], z_height[-1], 'ro', markersize=10, label='Last Point')

            # Mark leading edge
            ax1.plot(y_chord[min_y_idx], z_height[min_y_idx], 'co', markersize=10, label='Leading Edge')

            # Show direction with arrows
            for j in range(0, len(coords)-1, max(1, len(coords)//10)):
                ax1.annotate("", xy=(y_chord[j+1], z_height[j+1]), 
                            xytext=(y_chord[j], z_height[j]),
                            arrowprops=dict(arrowstyle="->", color='r', lw=1.5))

            # Check if first and last points match
            te_distance = np.linalg.norm(np.array([y_chord[0], z_height[0]]) - 
                                         np.array([y_chord[-1], z_height[-1]]))

            ax1.set_title(f"Layer {i}: {layer_name}\nTE Gap: {te_distance:.6f}")
            ax1.set_xlabel('Chord (Y)')
            ax1.set_ylabel('Height (Z)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Add point indices at regular intervals
            for j in range(0, len(coords), max(1, len(coords)//5)):
                ax1.annotate(f"{j}", (y_chord[j], z_height[j]),
                            xytext=(5, 5), textcoords='offset points')

            # 2. TOP VIEW - span distribution (X-Y plane)
            ax2 = axes[i][1]

            # Plot the points from top view
            ax2.plot(x_span, y_chord, 'b-', linewidth=1, alpha=0.7)
            ax2.scatter(x_span, y_chord, c=range(len(coords)), cmap='viridis', 
                       s=50, zorder=5, alpha=0.7)

            # Mark first and last points
            ax2.plot(x_span[0], y_chord[0], 'go', markersize=10, label='First Point')
            ax2.plot(x_span[-1], y_chord[-1], 'ro', markersize=10, label='Last Point')

            # Mark leading edge
            ax2.plot(x_span[min_y_idx], y_chord[min_y_idx], 'co', markersize=10, label='Leading Edge')

            ax2.set_title(f"Top View (X-Y) - Layer {i}")
            ax2.set_xlabel('Span (X)')
            ax2.set_ylabel('Chord (Y)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            # Add point indices at regular intervals
            for j in range(0, len(coords), max(1, len(coords)//5)):
                ax2.annotate(f"{j}", (x_span[j], y_chord[j]),
                            xytext=(5, 5), textcoords='offset points')

        # Print detailed stats
        print("\n==== AIRFOIL DEBUG INFORMATION ====")
        print(f"Number of layers: {n_layers}")

        for i, layer in enumerate(self.layers):
            coords = layer.coords
            if not coords:
                continue

            y_chord = [p[1] for p in coords]
            min_y_idx = y_chord.index(min(y_chord))
            max_y_idx = y_chord.index(max(y_chord))

            print(f"\nLayer {i} ({os.path.basename(layer.filename)}):")
            print(f"  Total points: {len(coords)}")
            print(f"  First point (idx 0): {coords[0]}")
            print(f"  Leading edge (idx {min_y_idx}): {coords[min_y_idx]}")
            if max_y_idx != 0 and max_y_idx != len(coords)-1:
                print(f"  Max Y point (idx {max_y_idx}): {coords[max_y_idx]}")
            print(f"  Last point (idx {len(coords)-1}): {coords[-1]}")

            # Check trailing edge closure
            te_distance = np.linalg.norm(np.array(coords[0]) - np.array(coords[-1]))
            if te_distance > 1e-6:
                print(f"  WARNING: Trailing edge not closed! Gap = {te_distance:.8f}")
            else:
                print(f"  Trailing edge properly closed: Gap = {te_distance:.8f}")

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file)
            print(f"Debug visualization saved to {output_file}")
        else:
            plt.show()

def calculate_bounding_box_trimesh(points):
    """
    Calculate the bounding box for a set of 3D points using trimesh.
    
    Args:
        points: List of [x, y, z] coordinates
    
    Returns:
        A trimesh object representing the bounding box
    """
    # Create a Trimesh object from the points
    cloud = trimesh.points.PointCloud(points)
    
    # Get the bounding box as a Trimesh object
    bounding_box = cloud.bounding_box
    return bounding_box

def save_stl(mesh, filename):
    """
    Save a trimesh object as an STL file.
    
    Args:
        mesh: A trimesh object
        filename: Output STL file name
    """
    mesh.export(filename)
    print(f"STL saved as '{filename}'")

def main():
    # Configuration
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    wing_stl_path = output_dir / "wing.stl"
    bounding_box_stl_path = output_dir / "bounding_box.stl"
    
    # Generate the wing
    param_array = [0, 0.0, 0.0, 0.0, 1.0]  # Example parameters
    wing = airfoilLayers(
        density=3, 
        wing_span=0.065 / 2, 
        wing_chord=0.02, 
        surface_degree_u=2, 
        surface_degree_v=1, 
        sample_resolution=60
    )
    wing.create_geometry_from_array(param_array, ["naca0012.dat"], str(wing_stl_path))
    print(f"Wing STL saved as '{wing_stl_path}'")
    
    # Collect all points from the wing surface
    all_points = []
    for layer in wing.layers:
        all_points.extend(layer.coords)
    
    # Create a Trimesh object for the wing
    wing_mesh = trimesh.Trimesh(vertices=all_points)
    
    # Calculate the bounding box
    bounding_box = calculate_bounding_box_trimesh(all_points)
    print(f"Bounding box dimensions: {bounding_box.extents}")
    
    # Save the wing STL
    save_stl(wing_mesh, str(wing_stl_path))
    
    # Save the bounding box STL
    save_stl(bounding_box, str(bounding_box_stl_path))

if __name__ == "__main__":
    main()

    # Example usage - Better parameter grouping
    airfoil_files = ["naca2412.dat"]
    # Format: [airfoil0, pitch0, y_offset0, z_offset0, scale0, airfoil1, pitch1, y_offset1, z_offset1, scale1, ...]
    param_array = [0, 5.0, 0.0, 0.0, 1.0, 0, 10.0, 0.0, 0.0, 0.8, 0, 15.0, 0.0, 0.0, 0.6]
    
    layers = airfoilLayers(density=3, wing_span=10.0, wing_chord=2.0)
    output_file = layers.create_geometry_from_array(param_array, airfoil_files, "example_optimized.stl")
    print(f"Generated {output_file}")
    
    # Static method example
    output_file2 = airfoilLayers.quick_generate(param_array, airfoil_files, density=3, 
                                               wing_span=10.0, wing_chord=2.0, 
                                               output_filename="quick_example.stl")
    print(f"Generated {output_file2}")