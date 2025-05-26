# Medical Image Co-Registration Project

This project implements a complete pipeline for 3D CT image co-registration and visualization, including DICOM processing, rigid registration with multiple optimization methods, and advanced visualization techniques.

## Features

- DICOM image loading and processing
- 3D rigid co-registration with quaternion-based transformation
- Multiple optimization methods (L-BFGS-B, Powell, Nelder-Mead)
- Compare optimization methods with metrics (MSE, MAE, MI)
- Maximum Intensity Projection (MIP) visualization
- Interactive contrast adjustment
- Segmentation mask overlay

## Project Structure

```
.
├── data/                      # Input data directory
│   ├── reference/            # Reference CT scan DICOM files
│   ├── input/               # Input CT scan DICOM files
│   └── masks/               # Segmentation mask files
├── output/                   # Generated visualizations
│   ├── reference_MIP/       # Reference MIP visualizations
│   ├── input_MIP/           # Input MIP visualizations
│   ├── registered_input_MIP/ # Registered input MIP visualizations
│   └── plots/               # Optimizer loss plots
├── main.py                  # Main execution script
├── coregistration.py        # Registration algorithms
├── visualizer.py            # Visualization utilities
├── utils.py                 # Helper functions
├── config.py               # Configuration settings
└── chooser.py              # Interactive contrast adjustment
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install numpy scipy matplotlib pillow pydicom tqdm
   ```

## Usage

1. Place your DICOM data in the respective directories:
   - Reference CT scan in `data/reference/`
   - Input CT scan in `data/input/`
   - Segmentation masks in `data/masks/`

2. Run the main script:
   ```bash
   python main.py
   ```

3. View the results in the `output/` directory:
   - Rotation animations as GIFs
   - Individual frames as PNG images
   - Registration performance metrics in console output

## Co-Registration Methods

The project implements and compares three optimization methods for 3D rigid registration:

1. **L-BFGS-B**: Limited-memory BFGS with box constraints
2. **Powell**: Derivative-free optimization algorithm
3. **Nelder-Mead**: Simplex-based optimization method

The registration uses a quaternion-based approach for rotation and translation parameters to align the input image with the reference image. Initial translation parameters are calculated based on the centers of mass of the images.

## Visualization Features

- **Maximum Intensity Projection (MIP)**: 128-step rotation sequences
- **Mask Overlays**: Semi-transparent visualization of segmentations


## Metrics

The following metrics are used to evaluate registration quality:

- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **MI**: Mutual Information

## Configuration

You can modify `config.py` to change:
- Input/output paths
- Visualization settings
- Animation parameters
