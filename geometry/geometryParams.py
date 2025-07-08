import numpy as np

class GeometryParams:
    def __init__(self, ts: list[float]):
        self.ts = ts
        self.fitness = None
        
    def clone(self):
        return GeometryParams(ts=self.ts.copy())