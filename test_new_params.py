#!/usr/bin/env python3

"""
Test the improved parameter generation and mutation
"""

import sys
sys.path.append('.')

from main import generate_initial_population
from config import *
import numpy as np

def test_new_generation():
    print("Testing new parameter generation...")
    
    # Generate a small population
    population = generate_initial_population(10, 5)
    
    print(f"Generated {len(population)} individuals")
    
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
            print(f"  {name:12s}: std={std_val:.6f}, range={range_val:.6f}")
    
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

if __name__ == "__main__":
    test_new_generation()
