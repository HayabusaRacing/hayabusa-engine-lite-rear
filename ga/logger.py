import os
import json

from config import RESULTS_DIR

class ResultSaver:
    def __init__(self, root_dir=RESULTS_DIR):
        self.root_dir = root_dir
        os.makedirs(root_dir, exist_ok=True)

    def save_individual(self, generation: int, child: int, ts: list[float], fitness_dict: dict):
        gen_dir = os.path.join(self.root_dir, f"generation{generation:03d}")
        child_dir = os.path.join(gen_dir, f"child{child:03d}")
        os.makedirs(child_dir, exist_ok=True)

        with open(os.path.join(child_dir, "ts.json"), "w") as f:
            json.dump(ts, f, indent=2)

        with open(os.path.join(child_dir, "fitness.json"), "w") as f:
            json.dump(fitness_dict, f, indent=2)