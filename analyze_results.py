#!/usr/bin/env python3

"""
Analyze parameter evolution in the GA results
"""

import json
import numpy as np
import os
from pathlib import Path

def analyze_results():
    results_dir = Path("results")
    
    if not results_dir.exists():
        print("No results directory found!")
        return
    
    # Find all generations
    generations = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("generation")])
    
    print(f"Found {len(generations)} generations")
    
    # Analyze parameter evolution
    param_evolution = {}
    fitness_evolution = []
    
    for gen_dir in generations:
        gen_num = int(gen_dir.name.replace("generation", ""))
        
        # Get all children in this generation
        children = sorted([d for d in gen_dir.iterdir() if d.is_dir() and d.name.startswith("child")])
        
        gen_params = []
        gen_fitness = []
        
        for child_dir in children:
            ts_file = child_dir / "ts.json"
            fitness_file = child_dir / "fitness.json"
            
            if ts_file.exists():
                with open(ts_file, 'r') as f:
                    ts_data = json.load(f)
                    gen_params.append(ts_data)
            
            if fitness_file.exists():
                with open(fitness_file, 'r') as f:
                    fitness_data = json.load(f)
                    if 'total_fitness' in fitness_data:
                        gen_fitness.append(fitness_data['total_fitness'])
                    elif isinstance(fitness_data, (int, float)):
                        gen_fitness.append(fitness_data)
        
        if gen_params:
            param_evolution[gen_num] = gen_params
            fitness_evolution.append({
                'generation': gen_num,
                'fitness_values': gen_fitness,
                'best': min(gen_fitness) if gen_fitness else float('inf'),
                'worst': max(gen_fitness) if gen_fitness else float('inf'),
                'mean': np.mean(gen_fitness) if gen_fitness else float('inf'),
                'std': np.std(gen_fitness) if gen_fitness else 0
            })
    
    print("\n=== FITNESS EVOLUTION ===")
    for gen_stats in fitness_evolution:
        print(f"Gen {gen_stats['generation']:2d}: Best={gen_stats['best']:.6f}, "
              f"Mean={gen_stats['mean']:.6f}, Std={gen_stats['std']:.6f}")
    
    # Analyze parameter diversity
    print(f"\n=== PARAMETER DIVERSITY ANALYSIS ===")
    
    if len(param_evolution) >= 2:
        first_gen = 0
        last_gen = max(param_evolution.keys())
        
        first_params = np.array(param_evolution[first_gen])
        last_params = np.array(param_evolution[last_gen])
        
        print(f"Parameter array length: {first_params.shape[1] if len(first_params) > 0 else 'N/A'}")
        print(f"Population size: {len(first_params)}")
        
        # Parameter names for analysis
        param_names = []
        for layer in range(1, 3):  # Assuming 2 outer layers (density=3, center fixed)
            param_names.extend([
                f"L{layer}_airfoil_type",
                f"L{layer}_pitch_angle", 
                f"L{layer}_y_offset",
                f"L{layer}_z_offset",
                f"L{layer}_scale"
            ])
        
        print(f"\nParameter diversity (Gen {first_gen} -> Gen {last_gen}):")
        
        for i, param_name in enumerate(param_names):
            if i < first_params.shape[1]:
                first_std = np.std(first_params[:, i])
                last_std = np.std(last_params[:, i])
                first_range = np.max(first_params[:, i]) - np.min(first_params[:, i])
                last_range = np.max(last_params[:, i]) - np.min(last_params[:, i])
                
                print(f"  {param_name:20s}: "
                      f"Std {first_std:.6f}->{last_std:.6f}, "
                      f"Range {first_range:.6f}->{last_range:.6f}")
        
        # Check for convergence issues
        print(f"\n=== CONVERGENCE ANALYSIS ===")
        converged_params = 0
        low_diversity_params = 0
        
        for i, param_name in enumerate(param_names):
            if i < last_params.shape[1]:
                std_val = np.std(last_params[:, i])
                range_val = np.max(last_params[:, i]) - np.min(last_params[:, i])
                
                if std_val < 0.001:  # Very low diversity
                    converged_params += 1
                    print(f"  CONVERGED: {param_name} (std={std_val:.6f})")
                elif std_val < 0.01:  # Low diversity
                    low_diversity_params += 1
                    print(f"  LOW DIVERSITY: {param_name} (std={std_val:.6f})")
        
        print(f"\nSummary:")
        print(f"  Converged parameters: {converged_params}/{len(param_names)}")
        print(f"  Low diversity parameters: {low_diversity_params}/{len(param_names)}")
        
        # Mutation effectiveness analysis
        print(f"\n=== MUTATION EFFECTIVENESS ===")
        
        # Compare consecutive generations
        if len(param_evolution) > 1:
            recent_gens = sorted(param_evolution.keys())[-2:]
            if len(recent_gens) == 2:
                gen1, gen2 = recent_gens
                params1 = np.array(param_evolution[gen1])
                params2 = np.array(param_evolution[gen2])
                
                # Handle different population sizes
                min_pop_size = min(len(params1), len(params2))
                params1 = params1[:min_pop_size]
                params2 = params2[:min_pop_size]
                
                total_change = 0
                for i in range(min(params1.shape[1], params2.shape[1])):
                    gen_change = np.mean(np.abs(params2[:, i] - params1[:, i]))
                    total_change += gen_change
                    if i < len(param_names):
                        print(f"  {param_names[i]:20s}: Avg change = {gen_change:.6f}")
                
                print(f"  Total parameter change: {total_change:.6f}")
                
                if total_change < 0.01:
                    print("  ⚠️  WARNING: Very low parameter change between generations!")
                    print("     Consider increasing mutation rate or strength.")

def suggest_hyperparameter_adjustments():
    print(f"\n=== HYPERPARAMETER SUGGESTIONS ===")
    
    # Current values
    print("Current hyperparameters:")
    print("  MUTATION_RATE = 0.4")
    print("  MUTATION_STRENGTH = 0.008") 
    print("  MUTATION_FACTOR = 0.2")
    
    print("\nSuggested adjustments for better exploration:")
    print("1. Increase mutation rate: MUTATION_RATE = 0.6-0.8")
    print("2. Increase mutation strength: MUTATION_STRENGTH = 0.02-0.05") 
    print("3. Increase mutation factor: MUTATION_FACTOR = 0.3-0.5")
    print("4. Consider adding crossover operations")
    print("5. Ensure initial population uses random values, not zeros!")

if __name__ == "__main__":
    analyze_results()
    suggest_hyperparameter_adjustments()
