from ga.individual import Individual
from ga.evaluate import evaluate
import random

class Optimizer:
    def __init__(self, population_size=10, generations=20):
        self.population_size = population_size
        self.generations = generations
        self.population = [Individual(ts=[0.5] * 300) for _ in range(population_size)]

    def run(self):
        for gen in range(self.generations):
            print(f"\n=== Generation {gen} ===")
            for individual in self.population:
                if individual.fitness is None:
                    individual.fitness = evaluate(individual.params)

            self.population.sort(key=lambda ind: ind.fitness)
            best = self.population[0]
            print(f"Best Cd: {best.fitness}")

            next_gen = self.population[:2]
            while len(next_gen) < self.population_size:
                parent1, parent2 = random.sample(self.population[:5], 2)
                child = parent1.crossover(parent2).mutate()
                next_gen.append(child)
            self.population = next_gen
