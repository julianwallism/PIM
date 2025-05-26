"""DICOM processing utilities."""

from typing import List, Tuple
import numpy as np
import highdicom as hd
from pydicom.dataset import Dataset
import pydicom
import os

def load_dicom_slices(path: str) -> List[Dataset]:
    """Load DICOM slices from a directory.
    
    Args:
        path: Path to the directory containing DICOM files
        
    Returns:
        List of DICOM datasets sorted by instance number
    """
    slices: list[pydicom.Dataset] = []
    sorted_files = sorted(os.listdir(path))
    for file in sorted_files:
        if file.endswith('.dcm'):
            file_path = os.path.join(path, file)
            slices.append(pydicom.dcmread(file_path))
    return sorted(slices, key=lambda x: x.InstanceNumber)

def create_3d_array_from_dicom(slices: List[Dataset]) -> np.ndarray:
    """Create a 3D numpy array from DICOM slices.
    
    Args:
        slices: List of DICOM datasets
        
    Returns:
        3D numpy array of shape (slices, height, width)
    """
    img_shape = slices[0].pixel_array.shape
    img_3d = np.zeros((len(slices), *img_shape), dtype=np.int16)
    for i, slice in enumerate(slices):
        img_3d[i] = slice.pixel_array
    return img_3d

def get_spacing_info(slice: Dataset) -> Tuple[float, float, float]:
    """Get pixel and slice spacing information from a DICOM slice.
    
    Args:
        slice: DICOM dataset
        
    Returns:
        Tuple of (pixel_spacing_x, pixel_spacing_y, slice_spacing)
    """
    pixel_spacing = slice.get("PixelSpacing")
    slice_spacing = slice.get("SpacingBetweenSlices")
    return pixel_spacing[0], pixel_spacing[1], slice_spacing

def load_segmentation_mask(mask_path: str) -> Tuple[np.ndarray, List[float]]:
    """Load segmentation mask from DICOM file.
    
    Args:
        mask_path: Path to the segmentation DICOM file
        
    Returns:
        Tuple of (mask array, z-positions)
    """
    seg = hd.seg.segread(mask_path)
    mask = seg.pixel_array
    
    z_positions = [
        float(frame.PlanePositionSequence[0].ImagePositionPatient[2])
        for frame in seg.PerFrameFunctionalGroupsSequence
    ]
    
    return mask, z_positions

def align_mask_with_ct(mask: np.ndarray, 
                      mask_z_positions: List[float],
                      ct_z_positions: List[float],
                      ct_shape: Tuple[int, int, int]) -> np.ndarray:
    """Align segmentation mask with CT volume.
    
    Args:
        mask: Segmentation mask array
        mask_z_positions: Z-positions of mask slices
        ct_z_positions: Z-positions of CT slices
        ct_shape: Shape of CT volume
        
    Returns:
        Aligned mask array of same shape as CT volume
    """
    full_mask = np.zeros(ct_shape, dtype=np.uint8)
    
    for i, z_pos in enumerate(mask_z_positions):
        if z_pos in ct_z_positions:
            ct_idx = ct_z_positions.index(z_pos)
            full_mask[ct_idx] = mask[i]
            
    return full_mask

def rotate_on_axial_plane(img: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    """Rotate the image on the axial plane.
    
    Args:
        img: 3D image array
        angle_in_degrees: Rotation angle in degrees
        
    Returns:
        Rotated image array
    """
    import scipy.ndimage
    return scipy.ndimage.rotate(img, angle_in_degrees, axes=(1, 2), reshape=False)

def MIP_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """Compute the maximum intensity projection on the sagittal orientation.
    
    Args:
        img_dcm: 3D image array
        
    Returns:
        2D array representing the maximum intensity projection
    """
    return np.max(img_dcm, axis=2)
