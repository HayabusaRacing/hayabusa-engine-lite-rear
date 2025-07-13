from ga.individual import Individual
from ga.logger import ResultSaver
from ga.evaluate import evaluate
from geometry.geometryParams import GeometryParams
from geometry.rayBundle import RayBundle
import numpy as np
import random

from config import MESH_WIDTH, MESH_HEIGHT, MESH_DEPTH, MESH_DENSITY, MESH_CENTER, MESH_UNIT, NUM_GENERATIONS, POPULATION_SIZE, MUTATION_FACTOR

def get_ts_length_from_raybundle():
    dummy = RayBundle(width=MESH_WIDTH, height=MESH_HEIGHT, depth=MESH_DEPTH, density=MESH_DENSITY, center=MESH_CENTER, unit=MESH_UNIT)
    return len(dummy.ts)

def dummy_evaluate(params):
    try:
        ts = np.array(params.ts)
        cd = np.mean(ts)
        return float(cd)
    except Exception as e:
        print(f"Evaluation error: {e}")
        return float('inf')

TS_LENGTH = get_ts_length_from_raybundle()

def generate_initial_population(ts_length, size):
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
    saver = ResultSaver()
    population = generate_initial_population(TS_LENGTH, POPULATION_SIZE)

    for generation in range(NUM_GENERATIONS):
        print(f"Generation {generation}")
        generation_fitness = []
        
        for i, indiv in enumerate(population):
            fitness = evaluate(indiv.params)
            indiv.fitness = fitness
            generation_fitness.append(fitness)
            
            fitness_dict = indiv.params.fitness_breakdown if indiv.params.fitness_breakdown else {"fitness": fitness}
            
            saver.save_individual(
                generation=generation,
                child=i,
                ts=indiv.params.ts,
                fitness_dict=fitness_dict
            )

        saver.save_generation_summary(generation, generation_fitness)
        saver.save_fitness_log()

        population.sort(key=lambda ind: ind.fitness)
        next_generation = [indiv.clone() for indiv in population[:POPULATION_SIZE // 2]]

        while len(next_generation) < POPULATION_SIZE:
            parent = random.choice(next_generation).clone()
            parent = parent.mutate(rate=MUTATION_FACTOR)
            next_generation.append(parent)

        population = next_generation

if __name__ == "__main__":
    evolve()
