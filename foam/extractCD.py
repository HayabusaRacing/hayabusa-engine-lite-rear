import os
import numpy as np

def extract_latest_cd(post_dir: str = "../case/postProcessing/forceCoeffs1/0"):
    time_dirs = [d for d in os.listdir(post_dir) if os.path.isdir(os.path.join(post_dir, d))]
    time_dirs = sorted(time_dirs, key=lambda x: float(x))
    latest_time = time_dirs[-1]

    filepath = os.path.join(post_dir, latest_time, "coefficient.dat")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"forceCoeffs.dat not found at: {filepath}")
    
    data = np.loadtxt(filepath, comments="#")
    latest_row = data[-1]
    time, Cl, Cd, Cm = latest_row[0], latest_row[2], latest_row[3], latest_row[4]

    return {
        "time": time,
        "Cl": Cl,
        "Cd": Cd,
        "Cm": Cm
    }

if __name__ == "__main__":
    result = extract_latest_cd()
    print(f"Cd = {result['Cd']}, Cl = {result['Cl']}, Cm = {result['Cm']} at time = {result['time']}")
