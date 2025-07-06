import shutil
import os

def setup_case(base_dir="../baseCase", case_dir="../case"):
    if os.path.exists(case_dir):
        shutil.rmtree(case_dir)

    shutil.copytree(base_dir, case_dir)
    print(f"Copied {base_dir} → {case_dir} Completed")

if __name__ == "__main__":
    setup_case()
