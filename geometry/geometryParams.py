import numpy as np

class GeometryParams:
    def __init__(self, ts: list[float]):
        self.ts = ts
        self.fitness = None
        self.fitness_breakdown = {}
        
    def clone(self):
        cloned = GeometryParams(ts=self.ts.copy())
        cloned.fitness = self.fitness
        cloned.fitness_breakdown = self.fitness_breakdown.copy() if self.fitness_breakdown else None
        return cloned