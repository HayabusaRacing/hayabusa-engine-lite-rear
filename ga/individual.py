from geometry.geometryParams import GeometryParams
import numpy as np
import random

from config import MUTATION_RATE, MUTATION_STRENGTH, MUTATION_SCALING

class Individual:
    def __init__(self, ts):
        self.params = GeometryParams(ts)
        self.fitness = None        

    def mutate(self, rate=MUTATION_RATE, strength=MUTATION_STRENGTH):
        new_ts = []
        for i, t in enumerate(self.params.ts):
            if random.random() < rate:
                # Determine parameter type and apply appropriate scaling
                param_index = i % 5  # 0=airfoil_type, 1=pitch, 2=y_offset, 3=z_offset, 4=scale
                
                if param_index == 1:  # pitch_angle
                    scaled_strength = strength * MUTATION_SCALING['pitch_angle']
                elif param_index == 2:  # y_offset
                    scaled_strength = strength * MUTATION_SCALING['y_offset']
                elif param_index == 3:  # z_offset
                    scaled_strength = strength * MUTATION_SCALING['z_offset']
                elif param_index == 4:  # scale
                    scaled_strength = strength * MUTATION_SCALING['scale']
                else:  # airfoil_type (will be overwritten anyway)
                    scaled_strength = strength
                
                t += np.random.normal(0, scaled_strength)
            new_ts.append(max(0, t))
        
        # Preserve airfoil types (multiples of 5)
        for i in range(0, len(new_ts), 5):
            new_ts[i] = self.params.ts[i]
        
        return Individual(new_ts)

    def crossover(self, other):
        ts1 = self.params.ts
        ts2 = other.params.ts
        child_ts = [(a + b) / 2 for a, b in zip(ts1, ts2)]
        return Individual(child_ts)
    
    def clone(self):
        return Individual(self.params.ts.copy())