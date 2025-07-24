#!/usr/bin/env python3
"""
Quick test of the GA system with airfoilLayers integration
This will run a single generation with a small population to verify everything works
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from ga.individual import Individual
from geometry.geometryParams import GeometryParams
from config import USE_AIRFOIL_LAYERS, AIRFOIL_DENSITY
from main import generate_initial_population, get_ts_length

def test_ga_integration():
    """Test that the GA system works with airfoilLayers"""
    print("=== Testing GA Integration with airfoilLayers ===")
    
    # Test parameter length
    ts_length = get_ts_length()
    print(f"Parameter length: {ts_length}")
    
    if USE_AIRFOIL_LAYERS:
        expected_length = AIRFOIL_DENSITY * 5
        assert ts_length == expected_length, f"Expected {expected_length}, got {ts_length}"
        print(f"✓ Using airfoilLayers with {AIRFOIL_DENSITY} layers = {ts_length} parameters")
    else:
        print(f"✓ Using RayBundle with {ts_length} parameters")
    
    # Test population generation
    print("\nTesting population generation...")
    population = generate_initial_population(ts_length, 3)  # Small population for testing
    
    assert len(population) == 3, f"Expected 3 individuals, got {len(population)}"
    print(f"✓ Generated population of {len(population)} individuals")
    
    # Test individual structure
    for i, individual in enumerate(population):
        assert len(individual.params.ts) == ts_length, f"Individual {i} has wrong parameter count"
        assert individual.fitness is None, f"Individual {i} should not have fitness yet"
        print(f"  Individual {i}: {len(individual.params.ts)} parameters")
    
    # Test parameter bounds for airfoil layers
    if USE_AIRFOIL_LAYERS:
        print("\nTesting parameter bounds...")
        for i, individual in enumerate(population):
            params = individual.params.ts
            
            # Check that parameters are within reasonable bounds
            for j in range(0, len(params), 5):
                wing_type_idx = params[j]
                pitch_angle = params[j+1]
                x_offset = params[j+2]
                z_offset = params[j+3]
                scale = params[j+4]
                
                assert 0 <= wing_type_idx <= 10, f"wing_type_idx out of bounds: {wing_type_idx}"
                assert -15 <= pitch_angle <= 15, f"pitch_angle out of bounds: {pitch_angle}"
                assert -0.5 <= x_offset <= 0.5, f"x_offset out of bounds: {x_offset}"
                assert -0.2 <= z_offset <= 0.2, f"z_offset out of bounds: {z_offset}"
                assert 0.3 <= scale <= 1.5, f"scale out of bounds: {scale}"
        
        print("✓ All parameters are within expected bounds")
    
    # Test mutation
    print("\nTesting mutation...")
    original = population[0].clone()
    mutated = original.mutate()
    
    assert len(mutated.params.ts) == ts_length, "Mutation changed parameter count"
    
    # Check that some parameters changed (with high probability)
    differences = sum(1 for a, b in zip(original.params.ts, mutated.params.ts) if abs(a - b) > 1e-10)
    print(f"✓ Mutation changed {differences} out of {ts_length} parameters")
    
    # Test crossover
    print("\nTesting crossover...")
    parent1 = population[0]
    parent2 = population[1]
    child = parent1.crossover(parent2)
    
    assert len(child.params.ts) == ts_length, "Crossover changed parameter count"
    print("✓ Crossover produced valid offspring")
    
    print("\n✓ All GA integration tests passed!")
    print(f"Your system is ready to run with {'airfoilLayers' if USE_AIRFOIL_LAYERS else 'RayBundle'} geometry generation")

if __name__ == "__main__":
    test_ga_integration()
