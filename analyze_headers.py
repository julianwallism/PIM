import pydicom
from pathlib import Path
import os

def analyze_dicom_headers(directory):
    """Analyze DICOM headers from a directory of DICOM files."""
    # Get first DICOM file in directory
    dicom_files = [f for f in os.listdir(directory) if f.endswith('.dcm')]
    if not dicom_files:
        print(f"No DICOM files found in {directory}")
        return
    
    # Load first slice
    first_slice = pydicom.dcmread(os.path.join(directory, dicom_files[0]))
    
    # Print relevant header information
    print(f"\nAnalyzing DICOM headers from {directory}")
    print("=" * 50)
    print(f"Patient Position: {getattr(first_slice, 'PatientPosition', 'N/A')}")
    print(f"Image Position (Patient): {getattr(first_slice, 'ImagePositionPatient', 'N/A')}")
    print(f"Image Orientation (Patient): {getattr(first_slice, 'ImageOrientationPatient', 'N/A')}")
    print(f"Pixel Spacing: {getattr(first_slice, 'PixelSpacing', 'N/A')}")
    print(f"Slice Thickness: {getattr(first_slice, 'SliceThickness', 'N/A')}")
    print(f"Modality: {getattr(first_slice, 'Modality', 'N/A')}")
    print(f"Manufacturer: {getattr(first_slice, 'Manufacturer', 'N/A')}")
    print(f"Study Description: {getattr(first_slice, 'StudyDescription', 'N/A')}")
    print(f"Series Description: {getattr(first_slice, 'SeriesDescription', 'N/A')}")
    print(f"Patient Position: {getattr(first_slice, 'PatientPosition', 'N/A')}")
    print(f"KVP: {getattr(first_slice, 'KVP', 'N/A')}")
    print(f"Exposure Time: {getattr(first_slice, 'ExposureTime', 'N/A')}")
    print(f"X-Ray Tube Current: {getattr(first_slice, 'XRayTubeCurrent', 'N/A')}")
    print(f"Window Center: {getattr(first_slice, 'WindowCenter', 'N/A')}")
    print(f"Window Width: {getattr(first_slice, 'WindowWidth', 'N/A')}")
    print(f"Rescale Intercept: {getattr(first_slice, 'RescaleIntercept', 'N/A')}")
    print(f"Rescale Slope: {getattr(first_slice, 'RescaleSlope', 'N/A')}")

def main():
    # Base directory from config
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    
    # Analyze reference and input DICOM headers
    reference_dir = data_dir / "reference"
    input_dir = data_dir / "input"
    
    analyze_dicom_headers(reference_dir)
    analyze_dicom_headers(input_dir)

if __name__ == "__main__":
    main() 