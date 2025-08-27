from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DIR = PROJECT_ROOT / "baseCase"
CASE_DIR = Path("/tmp/ramdisk/case")
TMP_DIR = Path("/tmp/ramdisk/tmp")
RESULTS_DIR = PROJECT_ROOT / "results"

POPULATION_SIZE = 50
NUM_GENERATIONS = 100
MUTATION_FACTOR = 0.4     # Increased from 0.2 for more aggressive evolution

PARALLEL_EVALUATIONS = 5
CORES_PER_CFD = 6

MUTATION_RATE = 0.7       # Increased from 0.4 for more exploration
MUTATION_STRENGTH = 0.025  # Increased from 0.008 for stronger mutations

# Parameter-specific mutation scaling (adjusted for stronger exploration)
MUTATION_SCALING = {
    'pitch_angle': 3.0,      # Increased from 2.0 - pitch angles can handle larger mutations
    'y_offset': 1.0,         # Increased from 0.5 - need more y-direction exploration  
    'z_offset': 0.8,         # Increased from 0.3 - need more z-direction exploration
    'scale': 1.5             # Increased from 1.0 - allow more scale variation
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
AIRFOIL_WING_SPAN = 0.050 / 2
AIRFOIL_WING_CHORD = 0.015
AIRFOIL_X_CENTER = 0
AIRFOIL_Y_CENTER = 0.0975
AIRFOIL_Z_CENTER = 0.05
AIRFOIL_SURFACE_DEGREE_U = 2  # Reduced from 3
AIRFOIL_SURFACE_DEGREE_V = 1  # Reduced from 2 - safer with only 3 layers
AIRFOIL_SAMPLE_RESOLUTION = 60  # Increased for better resolution
AIRFOIL_FILES = ["naca0012.dat", "e387.dat", "mh45.dat", "sd77032.dat"]  # Will be populated when you add more files

# Fixed end airfoil configuration (index density-1 - not optimized)
# This is the wing tip configuration that remains constant during optimization
AIRFOIL_END_FIXED = {
    'wing_type_idx': 0,      # Use first airfoil file
    'pitch_angle': 0.0,      # No pitch angle for end
    'y_offset': 0.0,         # No Y offset for end
    'z_offset': 0.0,         # No Z offset for end  
    'scale': 1.0             # Full scale for end
}

# Parameter bounds with individual min/max controls in physical units
PARAM_BOUNDS = {
    # Y-offset (chord direction) bounds in mm
    'Y_OFFSET_MIN_MM': -10.0,
    'Y_OFFSET_MAX_MM': 10.0,
    
    # Z-offset (height direction) bounds in mm
    'Z_OFFSET_MIN_MM': -2.0,
    'Z_OFFSET_MAX_MM': 10.0,
    
    # Pitch angle in degrees
    'PITCH_ANGLE_MIN': -5.0,
    'PITCH_ANGLE_MAX': 20.0,
    
    # Scale factor (unitless)
    'SCALE_MIN': 1,  # 15mm chord with 20mm base
    'SCALE_MAX': 1.6666666   # 25mm chord with 20mm base
}

# Bounding box limits (in model units)
BOUNDING_BOX_LIMITS = {
    'x_min': -0.05,  # Minimum x-coordinate
    'x_max': 0.05,   # Maximum x-coordinate
    'y_min': 0.089,   # Minimum y-coordinate (fixed - was incorrectly larger than y_max)
    'y_max': 0.126,    # Maximum y-coordinate (fixed - was incorrectly smaller than y_min)
    'z_min': 0.0475,    # Minimum z-coordinate
    'z_max': 0.065    # Maximum z-coordinate
}