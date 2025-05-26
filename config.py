"""Configuration settings for the MIP visualizer."""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"  # Point to data directory in project root

# DICOM paths
REFERENCE_DICOM_IMAGES_PATH = DATA_DIR / "reference"
INPUT_DICOM_PATH = DATA_DIR / "input"
TUMOR_MASK_PATH = DATA_DIR / "masks" / "21_PP_Ax5.00mm_ManualROI_Tumor.dcm"
LIVER_MASK_PATH = DATA_DIR / "masks" / "21_PP_Ax5.00mm_ManualROI_Liver.dcm"

# Output paths
REFERENCE_PROJECTION_OUTPUT_DIR = BASE_DIR / "output" / "reference_MIP"
REFERENCE_PROJECTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PROJECTION_OUTPUT_DIR = BASE_DIR / "output" / "input_MIP"
INPUT_PROJECTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REGISTERED_INPUT_PROJECTION_OUTPUT_DIR = BASE_DIR / "output" / "registered_input_MIP"
REGISTERED_INPUT_PROJECTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Visualization settings
FIGURE_SIZE = (13, 3)
COLORMAP = 'bone'
ANIMATION_DURATION = 100  # milliseconds
ROTATION_STEPS = 128
