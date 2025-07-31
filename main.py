from ga.individual import Individual
from ga.logger import ResultSaver
from ga.evaluate import evaluate
from ga.parallel_evaluate import evaluate_batch_parallel
from geometry.geometryParams import GeometryParams
from geometry.airfoilLayers import airfoilLayers
from utils.logging_utils import setup_logging
import numpy as np
import random

from config import (NUM_GENERATIONS, POPULATION_SIZE, MUTATION_FACTOR, PARALLEL_EVALUATIONS,
                   AIRFOIL_DENSITY, AIRFOIL_WING_SPAN, AIRFOIL_WING_CHORD,
                   AIRFOIL_Y_CENTER, AIRFOIL_X_CENTER, AIRFOIL_Z_CENTER,
                   AIRFOIL_SURFACE_DEGREE_U, AIRFOIL_SURFACE_DEGREE_V, AIRFOIL_SAMPLE_RESOLUTION,
                   AIRFOIL_CENTER_FIXED)

def get_ts_length():
    dummy = airfoilLayers(density=AIRFOIL_DENSITY, wing_span=AIRFOIL_WING_SPAN, wing_chord=AIRFOIL_WING_CHORD,
                         y_center=AIRFOIL_Y_CENTER, x_center=AIRFOIL_X_CENTER, z_center=AIRFOIL_Z_CENTER,
                         surface_degree_u=AIRFOIL_SURFACE_DEGREE_U, surface_degree_v=AIRFOIL_SURFACE_DEGREE_V,
                         sample_resolution=AIRFOIL_SAMPLE_RESOLUTION)
    return dummy.get_parameter_number()

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
                0,
                0.0,
                0.0,
                0.0,
                1.0
            ])
        population.append(Individual(ts))
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
