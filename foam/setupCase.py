import shutil
import os

from pathlib import Path
from config import BASE_DIR, CASE_DIR

def setup_case(base_dir=BASE_DIR, case_dir=CASE_DIR):
    if os.path.exists(case_dir):
        shutil.rmtree(case_dir)

    shutil.copytree(base_dir, case_dir)
    print(f"Copied {base_dir} → {case_dir} Completed")

if __name__ == "__main__":
    setup_case()
