from geometry.geometryParams import GeometryParams
import numpy as np
import random

from config import MUTATION_RATE, MUTATION_STRENGTH, MUTATION_SCALING, AIRFOIL_FILES

class Individual:
    def __init__(self, ts):
        self.params = GeometryParams(ts)
        self.fitness = None        

    def mutate(self, rate=MUTATION_RATE, strength=MUTATION_STRENGTH):
        new_ts = []
        for i, t in enumerate(self.params.ts):
            if random.random() < rate:
                param_index = i % 5  # 0=airfoil_type, 1=pitch, 2=y_offset, 3=z_offset, 4=scale

                # Special handling for airfoil type index (must be discrete)
                if param_index == 0:
                    # Either stay the same or randomly select a new airfoil type
                    if random.random() < 0.5:  # 50% chance to change airfoil
                        airfoil_count = len(AIRFOIL_FILES)
                        new_ts.append(random.randint(0, airfoil_count-1))
                    else:
                        new_ts.append(int(t))  # Keep current but ensure it's an integer
                else:
                    # Normal mutation for other parameters
                    scaled_strength = strength * MUTATION_SCALING.get(
                        ['wing_type', 'pitch_angle', 'y_offset', 'z_offset', 'scale'][param_index], 1.0
                    )
                    new_ts.append(t + np.random.normal(0, scaled_strength))
            else:
                new_ts.append(t)
        
        # Preserve airfoil types (multiples of 5)
        for i in range(0, len(new_ts), 5):
            new_ts[i] = self.params.ts[i]
        
        # Create new individual and ensure parameters are within bounds
        new_individual = Individual(new_ts)
        return new_individual.clip_to_bounds()

    def crossover(self, other):
        """Multiple-point crossover that preserves layer integrity"""
        ts1 = self.params.ts
        ts2 = other.params.ts
        
        params_per_layer = 5
        num_layers = len(ts1) // params_per_layer
        child_ts = []
        
        # For each layer, randomly choose which parent to inherit from
        for i in range(0, len(ts1), params_per_layer):
            # 50% chance to inherit each layer from either parent
            if random.random() < 0.5:
                child_ts.extend(ts1[i:i+params_per_layer])
            else:
                child_ts.extend(ts2[i:i+params_per_layer])
        
        # Create child and ensure parameters are within bounds
        child = Individual(child_ts)
        return child.clip_to_bounds()
    
    def clone(self):
        return Individual(self.params.ts.copy())

    def clip_to_bounds(self):
        """Clip parameter values to valid bounds"""
        from geometry.airfoilLayers import airfoilLayers
        from config import (AIRFOIL_DENSITY, AIRFOIL_WING_SPAN, AIRFOIL_WING_CHORD,
                           AIRFOIL_Y_CENTER, AIRFOIL_X_CENTER, AIRFOIL_Z_CENTER,
                           AIRFOIL_FILES)
        
        # Get bounds
        dummy = airfoilLayers(
            density=AIRFOIL_DENSITY,
            wing_span=AIRFOIL_WING_SPAN,
            wing_chord=AIRFOIL_WING_CHORD,
            y_center=AIRFOIL_Y_CENTER,
            x_center=AIRFOIL_X_CENTER,
            z_center=AIRFOIL_Z_CENTER
        )
        bounds = dummy.get_parameter_bounds()
        
        new_ts = []
        for i, t in enumerate(self.params.ts):
            param_index = i % 5  # 0=airfoil_type, 1=pitch, 2=y_offset, 3=z_offset, 4=scale
            param_type = ['wing_type_idx', 'pitch_angle', 'y_offset', 'z_offset', 'scale'][param_index]
            
            # Get bounds for this parameter type
            lower, upper = bounds[param_type]
            
            # Clip to bounds
            t_clipped = max(lower, min(t, upper))
            
            # For discrete parameters (airfoil type), convert to int
            if param_index == 0:
                t_clipped = int(t_clipped)
                
            new_ts.append(t_clipped)
        
        self.params = GeometryParams(new_ts)
        return self