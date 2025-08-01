#!/usr/bin/env python3

"""
Test the improved parameter generation and mutation (standalone)
"""

import sys
sys.path.append('.')

from ga.individual import Individual
from geometry.airfoilLayers import airfoilLayers
from config import *
import numpy as np
import random

def generate_test_population(size=5):
    # Generate parameters within appropriate bounds for airfoil layers
    dummy = airfoilLayers(density=AIRFOIL_DENSITY, wing_span=AIRFOIL_WING_SPAN, wing_chord=AIRFOIL_WING_CHORD,
                         y_center=AIRFOIL_Y_CENTER, x_center=AIRFOIL_X_CENTER, z_center=AIRFOIL_Z_CENTER)
    bounds = dummy.get_parameter_bounds()
    
    population = []
    for _ in range(size):
        ts = []
        # Only generate parameters for outer layers (center layer is fixed)
        # For density=3: generate parameters for layers 1 and 2 (skip layer 0)
        for i in range(1, AIRFOIL_DENSITY):
            # For each optimizable layer: [wing_type_idx, pitch_angle, y_offset, z_offset, scale]
            ts.extend([
                random.uniform(bounds['wing_type_idx'][0], bounds['wing_type_idx'][1]),
                random.uniform(bounds['pitch_angle'][0], bounds['pitch_angle'][1]),
                random.uniform(bounds['y_offset'][0], bounds['y_offset'][1]),
                random.uniform(bounds['z_offset'][0], bounds['z_offset'][1]),
                random.uniform(bounds['scale'][0], bounds['scale'][1])
            ])
        population.append(Individual(ts))
    return population, bounds

def test_new_generation():
    print("Testing new parameter generation...")
    
    # Generate a small population
    population, bounds = generate_test_population(5)
    
    print(f"Generated {len(population)} individuals")
    print(f"Parameter bounds: {bounds}")
    
    print("\nInitial population parameters:")
    for i, individual in enumerate(population):
        print(f"Individual {i}: {individual.params.ts}")
    
    # Check diversity
    all_params = np.array([ind.params.ts for ind in population])
    
    print(f"\nParameter diversity analysis:")
    param_names = ['L1_airfoil', 'L1_pitch', 'L1_y_offset', 'L1_z_offset', 'L1_scale',
                   'L2_airfoil', 'L2_pitch', 'L2_y_offset', 'L2_z_offset', 'L2_scale']
    
    for i, name in enumerate(param_names):
        if i < all_params.shape[1]:
            std_val = np.std(all_params[:, i])
            range_val = np.max(all_params[:, i]) - np.min(all_params[:, i])
            min_val = np.min(all_params[:, i])
            max_val = np.max(all_params[:, i])
            print(f"  {name:12s}: std={std_val:.6f}, range={range_val:.6f}, [{min_val:.6f}, {max_val:.6f}]")
    
    # Test mutation
    print(f"\nTesting mutation with new hyperparameters:")
    print(f"  MUTATION_RATE = {MUTATION_RATE}")
    print(f"  MUTATION_STRENGTH = {MUTATION_STRENGTH}")
    print(f"  MUTATION_SCALING = {MUTATION_SCALING}")
    
    original = population[0]
    mutated = original.mutate()
    
    print(f"\nOriginal:  {original.params.ts}")
    print(f"Mutated:   {mutated.params.ts}")
    
    changes = np.array(mutated.params.ts) - np.array(original.params.ts)
    print(f"Changes:   {changes}")
    print(f"Total abs change: {np.sum(np.abs(changes)):.6f}")
    
    # Test multiple mutations to see variation
    print(f"\nTesting mutation variation (10 mutations):")
    total_changes = []
    for i in range(10):
        mut = original.mutate()
        change = np.sum(np.abs(np.array(mut.params.ts) - np.array(original.params.ts)))
        total_changes.append(change)
        print(f"  Mutation {i+1}: total change = {change:.6f}")
    
    print(f"\nMutation statistics:")
    print(f"  Mean change: {np.mean(total_changes):.6f}")
    print(f"  Std change: {np.std(total_changes):.6f}")
    print(f"  Min change: {np.min(total_changes):.6f}")
    print(f"  Max change: {np.max(total_changes):.6f}")

if __name__ == "__main__":
    test_new_generation()
