#!/usr/bin/env python3
"""
Simple test to verify Individual mutation behavior
Tests that airfoil type parameters (multiples of 5) are preserved during mutation
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from ga.individual import Individual

def test_mutation():
    """Test mutation behavior with airfoil type preservation"""
    print("=== Individual Mutation Test ===")
    print()
    
    # Create initial individual with test parameters (10 parameters for 2 outer layers)
    initial_params = [
        0, 3.0, 0.005, 0.002, 0.95,  # Layer 1: wing_type=0, pitch=3°, y_offset, z_offset, scale
        1, 6.0, 0.010, 0.004, 0.90   # Layer 2: wing_type=1, pitch=6°, y_offset, z_offset, scale
    ]
    
    print(f"Initial parameters: {initial_params}")
    print(f"Airfoil types (indices 0, 5): {initial_params[0]}, {initial_params[5]}")
    print()
    
    individual = Individual(initial_params)
    
    # Test multiple mutations
    print("Testing 10 mutations:")
    print("Mutation | Airfoil Types | All Parameters")
    print("-" * 60)
    
    current_individual = individual
    for i in range(10):
        mutated = current_individual.mutate()
        airfoil_types = [mutated.params.ts[j] for j in range(0, len(mutated.params.ts), 5)]
        
        print(f"   {i+1:2d}    | {airfoil_types}     | {[round(x, 4) for x in mutated.params.ts]}")
        
        current_individual = mutated
    
    print()
    print("=== Verification ===")
    
    # Verify airfoil types are preserved
    final_airfoil_types = [current_individual.params.ts[j] for j in range(0, len(current_individual.params.ts), 5)]
    original_airfoil_types = [initial_params[j] for j in range(0, len(initial_params), 5)]
    
    if final_airfoil_types == original_airfoil_types:
        print("✅ SUCCESS: Airfoil types preserved across all mutations")
    else:
        print("❌ FAILURE: Airfoil types changed!")
        print(f"   Original: {original_airfoil_types}")
        print(f"   Final:    {final_airfoil_types}")
    
    # Check if other parameters changed
    other_params_original = [initial_params[j] for j in range(len(initial_params)) if j % 5 != 0]
    other_params_final = [current_individual.params.ts[j] for j in range(len(current_individual.params.ts)) if j % 5 != 0]
    
    if other_params_original != other_params_final:
        print("✅ SUCCESS: Non-airfoil parameters were mutated")
        changed_count = sum(1 for orig, final in zip(other_params_original, other_params_final) if abs(orig - final) > 1e-6)
        print(f"   {changed_count}/{len(other_params_original)} non-airfoil parameters changed")
    else:
        print("⚠️  WARNING: No non-airfoil parameters changed (might be due to low mutation rate)")
    
    return True

def test_crossover():
    """Quick test of crossover to make sure it still works"""
    print("\n=== Crossover Test ===")
    
    parent1_params = [0, 2.0, 0.005, 0.001, 0.95, 1, 4.0, 0.008, 0.003, 0.90]
    parent2_params = [0, 8.0, 0.015, 0.005, 0.85, 1, 12.0, 0.020, 0.007, 0.80]
    
    parent1 = Individual(parent1_params)
    parent2 = Individual(parent2_params)
    
    child = parent1.crossover(parent2)
    
    print(f"Parent 1: {[round(x, 3) for x in parent1_params]}")
    print(f"Parent 2: {[round(x, 3) for x in parent2_params]}")
    print(f"Child:    {[round(x, 3) for x in child.params.ts]}")
    
    # Check if child parameters are averages
    expected = [(a + b) / 2 for a, b in zip(parent1_params, parent2_params)]
    if all(abs(child.params.ts[i] - expected[i]) < 1e-6 for i in range(len(expected))):
        print("✅ SUCCESS: Crossover produces correct averages")
    else:
        print("❌ FAILURE: Crossover not working correctly")
    
    return True

def test_mutation_scaling():
    """Test that different parameter types get different mutation magnitudes"""
    print("\n=== Mutation Scaling Test ===")
    
    # Create individual with known values for comparison
    test_params = [0, 5.0, 0.010, 0.005, 1.0, 1, 10.0, 0.020, 0.010, 0.8]
    individual = Individual(test_params)
    
    # Collect mutation changes over many iterations
    pitch_changes = []
    y_offset_changes = []
    z_offset_changes = []
    scale_changes = []
    
    for _ in range(100):  # More iterations for better statistics
        mutated = individual.mutate()
        
        # Calculate absolute changes for each parameter type
        if mutated.params.ts[1] != test_params[1]:  # pitch changed
            pitch_changes.append(abs(mutated.params.ts[1] - test_params[1]))
        if mutated.params.ts[2] != test_params[2]:  # y_offset changed
            y_offset_changes.append(abs(mutated.params.ts[2] - test_params[2]))
        if mutated.params.ts[3] != test_params[3]:  # z_offset changed
            z_offset_changes.append(abs(mutated.params.ts[3] - test_params[3]))
        if mutated.params.ts[4] != test_params[4]:  # scale changed
            scale_changes.append(abs(mutated.params.ts[4] - test_params[4]))
    
    # Calculate average mutation magnitudes
    avg_pitch = sum(pitch_changes) / len(pitch_changes) if pitch_changes else 0
    avg_y_offset = sum(y_offset_changes) / len(y_offset_changes) if y_offset_changes else 0
    avg_z_offset = sum(z_offset_changes) / len(z_offset_changes) if z_offset_changes else 0
    avg_scale = sum(scale_changes) / len(scale_changes) if scale_changes else 0
    
    print(f"Average mutation magnitudes (100 iterations):")
    print(f"  Pitch angles: {avg_pitch:.4f}")
    print(f"  Y offsets:    {avg_y_offset:.4f}")
    print(f"  Z offsets:    {avg_z_offset:.4f}")
    print(f"  Scale:        {avg_scale:.4f}")
    
    # Verify scaling relationships
    if avg_pitch > avg_scale > avg_y_offset > avg_z_offset:
        print("✅ SUCCESS: Mutation scaling working correctly (pitch > scale > y_offset > z_offset)")
    else:
        print("⚠️  WARNING: Mutation scaling might need adjustment")
    
    return True

if __name__ == "__main__":
    test_mutation()
    test_crossover()
    test_mutation_scaling()
    print("\n=== Test Complete ===")
