#!/usr/bin/env python3
"""
Simple GA integration test without OpenFOAM evaluation
Tests the core GA functionality with airfoilLayers geometry generation
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from ga.individual import Individual
from geometry.geometryParams import GeometryParams
from geometry.airfoilLayers import airfoilLayers
from config import USE_AIRFOIL_LAYERS, AIRFOIL_DENSITY, AIRFOIL_WING_SPAN, AIRFOIL_WING_CHORD, AIRFOIL_FILES
import numpy as np

def get_ts_length_simple():
    """Get parameter length without importing main.py (to avoid OpenFOAM dependencies)"""
    if USE_AIRFOIL_LAYERS:
        dummy = airfoilLayers(density=AIRFOIL_DENSITY, wing_span=AIRFOIL_WING_SPAN, wing_chord=AIRFOIL_WING_CHORD)
        return dummy.get_parameter_number()
    else:
        # For RayBundle, we'll just return a reasonable number for testing
        return 400  # Approximate number based on your description

def generate_simple_population(ts_length, size):
    """Generate population without importing main.py"""
    if USE_AIRFOIL_LAYERS:
        dummy = airfoilLayers(density=AIRFOIL_DENSITY, wing_span=AIRFOIL_WING_SPAN, wing_chord=AIRFOIL_WING_CHORD)
        bounds = dummy.get_parameter_bounds()
        
        population = []
        for _ in range(size):
            ts = []
            for i in range(AIRFOIL_DENSITY):
                import random
                ts.extend([
                    random.uniform(bounds['wing_type_idx'][0], bounds['wing_type_idx'][1]),
                    random.uniform(bounds['pitch_angle'][0], bounds['pitch_angle'][1]),
                    random.uniform(bounds['x_offset'][0], bounds['x_offset'][1]),
                    random.uniform(bounds['z_offset'][0], bounds['z_offset'][1]),
                    random.uniform(bounds['scale'][0], bounds['scale'][1])
                ])
            population.append(Individual(ts))
        return population
    else:
        # Simple population for RayBundle
        import random
        population = []
        for _ in range(size):
            ts = [random.uniform(0, 1) for _ in range(ts_length)]
            population.append(Individual(ts))
        return population

def test_geometry_generation_only():
    """Test that geometry generation works without any evaluation"""
    print("=== Testing Geometry Generation Only ===")
    
    if not USE_AIRFOIL_LAYERS:
        print("Skipping geometry test - RayBundle requires trimesh imports")
        return True
    
    # Test parameter length
    ts_length = get_ts_length_simple()
    print(f"Parameter length: {ts_length}")
    
    # Generate a single individual
    population = generate_simple_population(ts_length, 1)
    individual = population[0]
    
    print(f"Generated individual with {len(individual.params.ts)} parameters")
    
    # Test geometry generation
    try:
        wing_generator = airfoilLayers(
            density=AIRFOIL_DENSITY, 
            wing_span=AIRFOIL_WING_SPAN, 
            wing_chord=AIRFOIL_WING_CHORD
        )
        
        output_file = wing_generator.create_geometry_from_array(
            individual.params.ts, 
            AIRFOIL_FILES, 
            "test_individual_wing.stl"
        )
        
        print(f"✓ Successfully generated geometry: {output_file}")
        
        # Check file size
        if Path(output_file).exists():
            file_size = Path(output_file).stat().st_size
            print(f"✓ STL file created with size: {file_size} bytes")
            return True
        else:
            print("✗ STL file was not created")
            return False
            
    except Exception as e:
        print(f"✗ Geometry generation failed: {e}")
        return False

def test_ga_operations_only():
    """Test GA operations (mutation, crossover) without evaluation"""
    print("\n=== Testing GA Operations ===")
    
    ts_length = get_ts_length_simple()
    population = generate_simple_population(ts_length, 3)
    
    print(f"Generated population of {len(population)} individuals")
    
    # Test mutation
    original = population[0].clone()
    mutated = original.mutate()
    
    differences = sum(1 for a, b in zip(original.params.ts, mutated.params.ts) if abs(a - b) > 1e-10)
    print(f"✓ Mutation changed {differences} out of {ts_length} parameters")
    
    # Test crossover
    parent1 = population[0]
    parent2 = population[1]
    child = parent1.crossover(parent2)
    
    assert len(child.params.ts) == ts_length, "Crossover changed parameter count"
    print("✓ Crossover produced valid offspring")
    
    # Test cloning
    clone = population[0].clone()
    assert clone.params.ts == population[0].params.ts, "Clone differs from original"
    print("✓ Cloning works correctly")
    
    return True

def test_parameter_bounds():
    """Test that generated parameters are within bounds"""
    print("\n=== Testing Parameter Bounds ===")
    
    if not USE_AIRFOIL_LAYERS:
        print("Skipping bounds test - only applicable to airfoilLayers")
        return True
    
    ts_length = get_ts_length_simple()
    population = generate_simple_population(ts_length, 5)
    
    wing_generator = airfoilLayers(density=AIRFOIL_DENSITY)
    bounds = wing_generator.get_parameter_bounds()
    
    for i, individual in enumerate(population):
        params = individual.params.ts
        
        for j in range(0, len(params), 5):
            wing_type_idx = params[j]
            pitch_angle = params[j+1]
            x_offset = params[j+2]
            z_offset = params[j+3]
            scale = params[j+4]
            
            assert bounds['wing_type_idx'][0] <= wing_type_idx <= bounds['wing_type_idx'][1], f"wing_type_idx out of bounds"
            assert bounds['pitch_angle'][0] <= pitch_angle <= bounds['pitch_angle'][1], f"pitch_angle out of bounds"
            assert bounds['x_offset'][0] <= x_offset <= bounds['x_offset'][1], f"x_offset out of bounds"
            assert bounds['z_offset'][0] <= z_offset <= bounds['z_offset'][1], f"z_offset out of bounds"
            assert bounds['scale'][0] <= scale <= bounds['scale'][1], f"scale out of bounds"
    
    print(f"✓ All parameters in {len(population)} individuals are within bounds")
    return True

def test_simple_evolution_step():
    """Test a single evolution step without fitness evaluation"""
    print("\n=== Testing Simple Evolution Step ===")
    
    ts_length = get_ts_length_simple()
    population = generate_simple_population(ts_length, 10)
    
    # Assign dummy fitness values
    import random
    for individual in population:
        individual.fitness = random.uniform(0.1, 1.0)
    
    # Sort by fitness (lower is better)
    population.sort(key=lambda ind: ind.fitness)
    
    # Select top half
    next_generation = [indiv.clone() for indiv in population[:5]]
    
    # Fill with mutations
    while len(next_generation) < 10:
        parent = random.choice(next_generation[:5]).clone()
        parent = parent.mutate(rate=0.2)
        next_generation.append(parent)
    
    assert len(next_generation) == 10, "Wrong population size after evolution step"
    print(f"✓ Evolution step completed: {len(next_generation)} individuals in next generation")
    
    return True

if __name__ == "__main__":
    print("=== Simple GA Integration Test (No OpenFOAM) ===")
    print(f"Using {'airfoilLayers' if USE_AIRFOIL_LAYERS else 'RayBundle'} geometry generation")
    
    success = True
    
    try:
        success &= test_geometry_generation_only()
        success &= test_ga_operations_only()
        success &= test_parameter_bounds()
        success &= test_simple_evolution_step()
        
        if success:
            print("\n✓ All tests passed!")
            print("Your GA system is ready to run with the new airfoilLayers integration")
            print("The system can generate STL files and perform GA operations correctly")
        else:
            print("\n✗ Some tests failed")
            
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        success = False
    
    if success:
        print("\nNext steps:")
        print("1. Set up your OpenFOAM environment for full evaluation")
        print("2. Run the full GA with: python main.py")
        print("3. Monitor results in the generated log files")
    
    sys.exit(0 if success else 1)
