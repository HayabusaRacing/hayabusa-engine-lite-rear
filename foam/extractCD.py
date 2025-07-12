import os
import numpy as np

from pathlib import Path
from config import CASE_DIR

def extract_latest_cd(post_dir: str = str(CASE_DIR / "postProcessing" / "forceCoeffs1" / "0")):
    filepath = os.path.join(post_dir, "coefficient.dat")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"coefficient.dat not found at: {filepath}")

    data = np.loadtxt(filepath, comments="#")
    latest_row = data[-1]
    time, Cd, Cl, Cm = latest_row[0], latest_row[1], latest_row[4], latest_row[6]

    return {
        "time": time,
        "Cd": Cd,
        "Cl": Cl,
        "Cm": Cm
    }

if __name__ == "__main__":
    result = extract_latest_cd()
    print(f"Cd = {result['Cd']}, Cl = {result['Cl']}, Cm = {result['Cm']} at time = {result['time']}")
