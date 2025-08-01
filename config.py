from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DIR = PROJECT_ROOT / "baseCase"
CASE_DIR = Path("/tmp/ramdisk/case")
TMP_DIR = Path("/tmp/ramdisk/tmp")
RESULTS_DIR = PROJECT_ROOT / "results"

POPULATION_SIZE = 50
NUM_GENERATIONS = 100
MUTATION_FACTOR = 0.2

PARALLEL_EVALUATIONS = 5
CORES_PER_CFD = 6

MUTATION_RATE = 0.4
MUTATION_STRENGTH = 0.008

# Parameter-specific mutation scaling
MUTATION_SCALING = {
    'pitch_angle': 2.0,      # Pitch angles can handle larger mutations (indices 1, 6, 11...)
    'y_offset': 0.5,         # Y offsets need smaller mutations (indices 2, 7, 12...)
    'z_offset': 0.3,         # Z offsets need even smaller mutations (indices 3, 8, 13...)
    'scale': 1.0             # Scale factors use base strength (indices 4, 9, 14...)
}

MIN_WING_DIMENSIONS = {
    'x': 0.01,
    'y': 0.005,
    'z': 0.002
}

DIMENSION_PENALTY_WEIGHT = 0.0
VOLUME_REWARD_WEIGHT = 0.0
SMOOTHNESS_REWARD_WEIGHT = 0.0
CD_WEIGHT = 1.0
EXPECTED_VOLUME_RANGE = [1e-6, 1e-4]

# Airfoil Layers Configuration
AIRFOIL_DENSITY = 3
AIRFOIL_WING_SPAN = 0.065 / 2
AIRFOIL_WING_CHORD = 0.02
AIRFOIL_X_CENTER = 0
AIRFOIL_Y_CENTER = -0.1075
AIRFOIL_Z_CENTER = 0.018
AIRFOIL_SURFACE_DEGREE_U = 3
AIRFOIL_SURFACE_DEGREE_V = 2
AIRFOIL_SAMPLE_RESOLUTION = 40
AIRFOIL_FILES = ["lorentz_teardrop.dat"]  # Will be populated when you add more files

# Fixed center airfoil configuration (index 0 - not optimized)
AIRFOIL_CENTER_FIXED = {
    'wing_type_idx': 0,      # Use first airfoil file
    'pitch_angle': 0.0,      # No pitch angle for center
    'y_offset': 0.0,         # No Y offset for center
    'z_offset': 0.0,         # No Z offset for center  
    'scale': 1.0             # Full scale for center
}