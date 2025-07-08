import os
import json
import numpy as np
from typing import Union, List, Dict

from config import RESULTS_DIR

class ResultSaver:
    def __init__(self, root_dir: str = RESULTS_DIR):
        self.root_dir = root_dir
        os.makedirs(root_dir, exist_ok=True)

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

        fitness_dict = convert_numpy_types(fitness_dict)

        try:
            with open(os.path.join(child_dir, "ts.json"), "w") as f:
                json.dump(ts, f, indent=2)
            with open(os.path.join(child_dir, "fitness.json"), "w") as f:
                json.dump(fitness_dict, f, indent=2)
        except (IOError, OSError) as e:
            print(f"Error saving individual (gen={generation}, child={child}): {e}")
            raise