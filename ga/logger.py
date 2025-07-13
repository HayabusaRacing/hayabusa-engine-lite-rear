import os
import json
import numpy as np
from typing import Union, List, Dict

from config import RESULTS_DIR

class ResultSaver:
    def __init__(self, root_dir = RESULTS_DIR):
        self.root_dir = str(root_dir)
        os.makedirs(self.root_dir, exist_ok=True)
        self.fitness_log = []
        self.fitness_log_path = os.path.join(self.root_dir, "fitnesses.json")

    def save_individual(self, generation: int, child: int, ts: Union[List[float], np.ndarray], fitness_dict: Dict):
        gen_dir = os.path.join(self.root_dir, f"generation{generation:03d}")
        child_dir = os.path.join(gen_dir, f"child{child:03d}")
        os.makedirs(child_dir, exist_ok=True)

        if isinstance(ts, np.ndarray):
            ts = ts.tolist()

        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj

        fitness_dict_converted = convert_numpy_types(fitness_dict)
        if not isinstance(fitness_dict_converted, dict):
            raise TypeError(f"fitness_dict must be a dict after conversion, got {type(fitness_dict_converted)}")
        fitness_dict = fitness_dict_converted

        try:
            with open(os.path.join(child_dir, "ts.json"), "w") as f:
                json.dump(ts, f, indent=2)
            with open(os.path.join(child_dir, "fitness.json"), "w") as f:
                json.dump(fitness_dict, f, indent=2)
                
            fitness_entry = {
                "generation": generation,
                "child": child,
                **fitness_dict
            }
            self.fitness_log.append(fitness_entry)
            
        except (IOError, OSError) as e:
            print(f"Error saving individual (gen={generation}, child={child}): {e}")
            raise

    def save_fitness_log(self):
        try:
            with open(self.fitness_log_path, "w") as f:
                json.dump(self.fitness_log, f, indent=2)
        except (IOError, OSError) as e:
            print(f"Error saving fitness log: {e}")
            raise

    def save_generation_summary(self, generation: int, population_fitness: List[float]):
        summary = {
            "generation": generation,
            "best_fitness": min(population_fitness),
            "worst_fitness": max(population_fitness),
            "mean_fitness": np.mean(population_fitness),
            "std_fitness": np.std(population_fitness)
        }
        
        summary_path = os.path.join(self.root_dir, f"generation{generation:03d}", "summary.json")
        try:
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
        except (IOError, OSError) as e:
            print(f"Error saving generation summary: {e}")
            raise