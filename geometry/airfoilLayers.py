from geomdl import BSpline
from geomdl import utilities
from geomdl import exchange
from geomdl.visualization import VisMPL
import numpy as np
import struct
import math

class geometryParameter:
    def __init__(self, wing_type, pitch_angle=0.0, x_offset=0.0, z_offset=0.0, scale=1.0):
        self.wing_type = wing_type
        self.pitch_angle = pitch_angle
        self.x_offset = x_offset
        self.z_offset = z_offset
        self.scale = scale


class airfoilLayer:
    def __init__(self, filename, x_offset=0.0, y_offset=0.0, z_offset=0.0, scale=1.0):
        self.filename = filename
        self.x_offset = x_offset
        self.y_offset = y_offset
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
                coords.append([x * self.scale + self.x_offset, self.y_offset, z * self.scale + self.z_offset])
        return coords
    
    def rotate_pitch(self, angle):
        angle_rad = math.radians(angle)
        cos_angle, sin_angle = math.cos(angle_rad), math.sin(angle_rad)
        for pt in self.coords:
            x, y, z = pt
            pt[0] = x * cos_angle - z * sin_angle
            pt[2] = x * sin_angle + z * cos_angle
        return self.coords
    
    def apply_transformation(self, transformation_matrix):
        for i, pt in enumerate(self.coords):
            pt_homogeneous = np.array([pt[0], pt[1], pt[2], 1.0])
            transformed = transformation_matrix @ pt_homogeneous
            self.coords[i] = transformed[:3].tolist()
        return self.coords
    
class airfoilLayers:
    def __init__(self, density, x_center=0.0, y_center=0.0, z_center=0.0, wing_span=0.0, wing_chord=0.0, 
                 surface_degree_u=3, surface_degree_v=2, sample_resolution=40):
        self.x_center = x_center
        self.y_center = y_center
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
            y_offset = self.y_center + (self.wing_span / (self.density - 1)) * i
            layer = self._create_layer(param, y_offset)
            layer.rotate_pitch(param.pitch_angle)
            self.add_layer(layer)
            
            if i != 0:
                negative_y_offset = self.y_center - (self.wing_span / (self.density - 1)) * i
                symmetric_layer = self._create_layer(param, negative_y_offset)
                symmetric_layer.rotate_pitch(param.pitch_angle)
                self.add_layer(symmetric_layer, 0)
    
    def _validate_parameters(self, parameters):
        if not all(isinstance(param, geometryParameter) for param in parameters):
            raise TypeError("All items must be geometryParameter instances")
        if len(parameters) != self.density:
            raise ValueError(f"Expected {self.density} parameters, got {len(parameters)}")
    
    def _create_layer(self, param, y_offset):
        return airfoilLayer(param.wing_type, x_offset=param.x_offset, y_offset=y_offset, 
                          z_offset=param.z_offset, scale=self.wing_chord * param.scale)
    
    def _create_surface(self):
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
        return self.density * 5

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
        
        if len(param_array) != self.density * param_per_layer:
            raise ValueError(f"Expected {self.density * param_per_layer} parameters, got {len(param_array)}")
        
        for i in range(self.density):
            base_idx = i * param_per_layer
            wing_type_idx = int(param_array[base_idx])
            wing_type = airfoil_files[wing_type_idx % len(airfoil_files)]
            
            pitch_angle = param_array[base_idx + 1]
            x_offset = param_array[base_idx + 2] 
            z_offset = param_array[base_idx + 3]
            scale = param_array[base_idx + 4]
            
            parameters.append(geometryParameter(wing_type, pitch_angle, x_offset, z_offset, scale))
        
        return parameters
    
    def get_parameter_bounds(self):
        return {
            'wing_type_idx': (0, 10),
            'pitch_angle': (-15, 15),
            'x_offset': (-0.5, 0.5),
            'z_offset': (-0.2, 0.2),
            'scale': (0.3, 1.5)
        }
    
    def get_array_size(self):
        return self.density * 5

    @staticmethod
    def quick_generate(param_array, airfoil_files, density=3, wing_span=1.0, wing_chord=0.5, output_filename="wing.stl"):
        wing = airfoilLayers(density=density, wing_span=wing_span, wing_chord=wing_chord)
        return wing.create_geometry_from_array(param_array, airfoil_files, output_filename)

if __name__ == "__main__":
    # Example usage - Better parameter grouping
    airfoil_files = ["naca2412.dat"]
    # Format: [airfoil0, pitch0, x_offset0, z_offset0, scale0, airfoil1, pitch1, x_offset1, z_offset1, scale1, ...]
    param_array = [0, 5.0, 0.0, 0.0, 1.0, 0, 10.0, 0.0, 0.0, 0.8, 0, 15.0, 0.0, 0.0, 0.6]
    
    layers = airfoilLayers(density=3, wing_span=10.0, wing_chord=2.0)
    output_file = layers.create_geometry_from_array(param_array, airfoil_files, "example_optimized.stl")
    print(f"Generated {output_file}")
    
    # Static method example
    output_file2 = airfoilLayers.quick_generate(param_array, airfoil_files, density=3, 
                                               wing_span=10.0, wing_chord=2.0, 
                                               output_filename="quick_example.stl")
    print(f"Generated {output_file2}")