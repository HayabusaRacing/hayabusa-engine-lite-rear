from ga.individual import Individual
from ga.logger import ResultSaver
from ga.evaluate import evaluate
from ga.parallel_evaluate import evaluate_batch_parallel
from geometry.geometryParams import GeometryParams
from geometry.rayBundle import RayBundle
from geometry.airfoilLayers import airfoilLayers
from utils.logging_utils import setup_logging
import numpy as np
import random

from config import (MESH_WIDTH, MESH_HEIGHT, MESH_DEPTH, MESH_DENSITY, MESH_CENTER, MESH_UNIT, 
                   NUM_GENERATIONS, POPULATION_SIZE, MUTATION_FACTOR, PARALLEL_EVALUATIONS,
                   USE_AIRFOIL_LAYERS, AIRFOIL_DENSITY, AIRFOIL_WING_SPAN, AIRFOIL_WING_CHORD,
                   AIRFOIL_SURFACE_DEGREE_U, AIRFOIL_SURFACE_DEGREE_V, AIRFOIL_SAMPLE_RESOLUTION)

def get_ts_length_from_raybundle():
    dummy = RayBundle(width=MESH_WIDTH, height=MESH_HEIGHT, depth=MESH_DEPTH, density=MESH_DENSITY, center=MESH_CENTER, unit=MESH_UNIT)
    return len(dummy.ts)

def get_ts_length_from_airfoil_layers():
    dummy = airfoilLayers(density=AIRFOIL_DENSITY, wing_span=AIRFOIL_WING_SPAN, wing_chord=AIRFOIL_WING_CHORD,
                         surface_degree_u=AIRFOIL_SURFACE_DEGREE_U, surface_degree_v=AIRFOIL_SURFACE_DEGREE_V,
                         sample_resolution=AIRFOIL_SAMPLE_RESOLUTION)
    return dummy.get_parameter_number()

def get_ts_length():
    if USE_AIRFOIL_LAYERS:
        return get_ts_length_from_airfoil_layers()
    else:
        return get_ts_length_from_raybundle()

def dummy_evaluate(params):
    try:
        ts = np.array(params.ts)
        cd = np.mean(ts)
        return float(cd)
    except Exception as e:
        print(f"Evaluation error: {e}")
        return float('inf')

TS_LENGTH = get_ts_length()

def generate_initial_population(ts_length, size):
    if USE_AIRFOIL_LAYERS:
        # For airfoil layers, generate parameters within appropriate bounds
        dummy = airfoilLayers(density=AIRFOIL_DENSITY, wing_span=AIRFOIL_WING_SPAN, wing_chord=AIRFOIL_WING_CHORD)
        bounds = dummy.get_parameter_bounds()
        
        population = []
        for _ in range(size):
            ts = []
            for i in range(AIRFOIL_DENSITY):
                # For each layer: [wing_type_idx, pitch_angle, x_offset, z_offset, scale]
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
        # Original RayBundle method
        dummy = RayBundle(width=MESH_WIDTH, height=MESH_HEIGHT, depth=MESH_DEPTH, density=MESH_DENSITY, center=MESH_CENTER, unit=MESH_UNIT)
        base_ts = dummy.ts
    
    population = []
    for i in range(size):
        if i == 0:
            varied_ts = base_ts.copy()
        else:
            varied_ts = []
            for t in base_ts:
                variation = np.random.normal(0, 0.001)
                varied_ts.append(max(0, t + variation))
        
        population.append(Individual(varied_ts))
    
    return population

def evolve():
    main_logger, _ = setup_logging()
    main_logger.info("Starting evolution process")
    
    saver = ResultSaver()
    population = generate_initial_population(TS_LENGTH, POPULATION_SIZE)
    main_logger.info(f"Generated initial population of {len(population)} individuals")

    for generation in range(NUM_GENERATIONS):
        main_logger.info(f"=== Generation {generation} ===")
        generation_fitness = []
        
        for batch_start in range(0, len(population), PARALLEL_EVALUATIONS):
            batch_end = min(batch_start + PARALLEL_EVALUATIONS, len(population))
            batch = population[batch_start:batch_end]
            
            main_logger.info(f"Evaluating batch {batch_start//PARALLEL_EVALUATIONS + 1}: individuals {batch_start}-{batch_end-1}")
            
            batch_params = [indiv.params for indiv in batch]
            batch_fitness = evaluate_batch_parallel(batch_params)
            
            for i, (indiv, fitness) in enumerate(zip(batch, batch_fitness)):
                indiv.fitness = fitness
                generation_fitness.append(fitness)
                
                fitness_dict = indiv.params.fitness_breakdown if indiv.params.fitness_breakdown else {"fitness": fitness}
                
                saver.save_individual(
                    generation=generation,
                    child=batch_start + i,
                    ts=indiv.params.ts,
                    fitness_dict=fitness_dict
                )

        saver.save_generation_summary(generation, generation_fitness)
        saver.save_fitness_log()
        
        best_fitness = min(generation_fitness)
        worst_fitness = max(generation_fitness)
        mean_fitness = np.mean(generation_fitness)
        main_logger.info(f"Generation {generation} summary - Best: {best_fitness:.6f}, Worst: {worst_fitness:.6f}, Mean: {mean_fitness:.6f}")

        population.sort(key=lambda ind: ind.fitness)
        next_generation = [indiv.clone() for indiv in population[:POPULATION_SIZE // 2]]

        while len(next_generation) < POPULATION_SIZE:
            parent = random.choice(next_generation).clone()
            parent = parent.mutate(rate=MUTATION_FACTOR)
            next_generation.append(parent)

        population = next_generation
        main_logger.info(f"Generated next generation with {len(population)} individuals")
    
    main_logger.info("Evolution process completed")

if __name__ == "__main__":
    evolve()
