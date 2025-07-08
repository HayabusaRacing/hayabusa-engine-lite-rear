from geometry.geometryParams import GeometryParams
import numpy as np
import random

class Individual:
    def __init__(self, ts):
        self.params = GeometryParams(ts)
        self.fitness = None

    def mutate(self, rate=0.1, strength=0.05):
        new_ts = []
        for t in self.params.ts:
            if random.random() < rate:
                t += np.random.normal(0, strength)
            new_ts.append(max(0, t))
        return Individual(new_ts)

    def crossover(self, other):
        ts1 = self.params.ts
        ts2 = other.params.ts
        child_ts = [(a + b) / 2 for a, b in zip(ts1, ts2)]
        return Individual(child_ts)