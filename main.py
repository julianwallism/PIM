"""Main script for MIP visualization and coregistration."""
from config import (
    REFERENCE_DICOM_IMAGES_PATH,
    INPUT_DICOM_PATH,
    TUMOR_MASK_PATH,
    LIVER_MASK_PATH,
    INPUT_PROJECTION_OUTPUT_DIR,
    REFERENCE_PROJECTION_OUTPUT_DIR,
    REGISTERED_INPUT_PROJECTION_OUTPUT_DIR,
    ROTATION_STEPS,
    ANIMATION_DURATION
)
from utils import (
    load_dicom_slices,
    create_3d_array_from_dicom,
    get_spacing_info,
    load_segmentation_mask,
    align_mask_with_ct
)
from coregistration import rigid_3D_coregistration, compare_optimization_methods
from visualizer import create_animation

from matplotlib.colors import ListedColormap
import numpy as np
from scipy import ndimage

def main():
    """Main function to process DICOM data and create visualizations."""
    
    #########################################################
    # Reference image
    #########################################################
    
    ref_slices = load_dicom_slices(str(REFERENCE_DICOM_IMAGES_PATH))
    print(f"Number of reference slices: {len(ref_slices)}")
        
    ref_img_dcm = create_3d_array_from_dicom(ref_slices)
    print(f"Reference 3D image shape: {ref_img_dcm.shape}")

    ref_pixel_spacing_x, ref_pixel_spacing_y, ref_slice_spacing = get_spacing_info(ref_slices[0])
    print(f"Pixel spacing: ({ref_pixel_spacing_x}, {ref_pixel_spacing_y})")
    print(f"Slice spacing: {ref_slice_spacing}")

    ref_ct_z_positions = [float(slice_obj.ImagePositionPatient[2]) for slice_obj in ref_slices] 

    mask, mask_z_positions = load_segmentation_mask(str(TUMOR_MASK_PATH))
    tumor_full_mask = align_mask_with_ct(mask, mask_z_positions, ref_ct_z_positions, ref_img_dcm.shape)
    
    mask, mask_z_positions = load_segmentation_mask(str(LIVER_MASK_PATH))
    liver_full_mask = align_mask_with_ct(mask, mask_z_positions, ref_ct_z_positions, ref_img_dcm.shape)

    create_animation(
        img=ref_img_dcm,
        masks={
            'tumor': tumor_full_mask,
            'liver': liver_full_mask
        },
        colormaps={
            'tumor': ListedColormap([(0,0,0,0), (1,0,0,0.3)]),
            'liver': ListedColormap([(0,0,0,0), (0,1,0,0.3)])
        },
        pixel_spacing=(ref_pixel_spacing_x, ref_pixel_spacing_y),
        slice_spacing=ref_slice_spacing,
        output_dir=str(REFERENCE_PROJECTION_OUTPUT_DIR),
        output_prefix='Animation',
        n_steps=ROTATION_STEPS,
        duration=ANIMATION_DURATION,
        contrast_window=(0, 400)
    )
    print("")
    
    #########################################################
    # Input image (unregistered)
    #########################################################
    
    # Load input DICOM
    input_slices = load_dicom_slices(str(INPUT_DICOM_PATH))
    input_slices = input_slices[:len(ref_slices)]
    print(f"Number of input slices: {len(input_slices)}")
    
    input_img_dcm = create_3d_array_from_dicom(input_slices)
    print(f"Input 3D image shape: {input_img_dcm.shape}")

    input_pixel_spacing_x, input_pixel_spacing_y, input_slice_spacing = get_spacing_info(input_slices[0])
    print(f"Pixel spacing: ({input_pixel_spacing_x}, {input_pixel_spacing_y})")
    print(f"Slice spacing: {input_slice_spacing}")

    create_animation(
        img=input_img_dcm,
        pixel_spacing=(input_pixel_spacing_x, input_pixel_spacing_y),
        masks={},
        colormaps={},
        slice_spacing=input_slice_spacing,
        output_dir=str(INPUT_PROJECTION_OUTPUT_DIR),
        output_prefix='Animation',
        n_steps=ROTATION_STEPS,
        duration=ANIMATION_DURATION,
        contrast_window=(-100, 1000)
    )
    print("")

    #########################################################
    # Coregistration
    #########################################################
    

    print("\nComparing all optimization methods...")
    registered_input_img_dcm, T = compare_optimization_methods(
        reference_img=ref_img_dcm, 
        input_img=input_img_dcm,
        maxiter=10,
        maxfun=500
    )

    
    print(f"Registered input 3D image shape: {registered_input_img_dcm.shape}")
    print(f"Transformation matrix: \n {T}")

    registered_input_pixel_spacing_x, registered_input_pixel_spacing_y, registered_input_slice_spacing = get_spacing_info(input_slices[0])
    print(f"Pixel spacing: ({registered_input_pixel_spacing_x}, {registered_input_pixel_spacing_y})")
    print(f"Slice spacing: {registered_input_slice_spacing}")

    input_ct_z_positions = [float(slice_obj.ImagePositionPatient[2]) for slice_obj in input_slices] 

    # Load and align masks with input CT positions
    mask, mask_z_positions = load_segmentation_mask(str(TUMOR_MASK_PATH))    
    tumor_full_mask = align_mask_with_ct(mask, mask_z_positions, input_ct_z_positions, input_img_dcm.shape)
    
    mask, mask_z_positions = load_segmentation_mask(str(LIVER_MASK_PATH))
    liver_full_mask = align_mask_with_ct(mask, mask_z_positions, input_ct_z_positions, input_img_dcm.shape)
    
    inv_rotation = np.linalg.inv(T[:3, :3])
    offset = -np.dot(inv_rotation, T[:3, 3])
    
    # Transform the masks using the same parameters as the image
    registered_tumor_mask = ndimage.affine_transform(
        tumor_full_mask,
        matrix=inv_rotation,
        offset=offset,
        output_shape=registered_input_img_dcm.shape,
        order=1
    )
    
    registered_liver_mask = ndimage.affine_transform(
        liver_full_mask,
        matrix=inv_rotation,
        offset=offset,
        output_shape=registered_input_img_dcm.shape,
        order=1
    )
    
    create_animation(
        img=registered_input_img_dcm,
        masks={
            'tumor': registered_tumor_mask,
            'liver': registered_liver_mask
        },
        colormaps={
            'tumor': ListedColormap([(0,0,0,0), (1,0,0,0.3)]),
            'liver': ListedColormap([(0,0,0,0), (0,1,0,0.3)])
        },
        pixel_spacing=(ref_pixel_spacing_x, ref_pixel_spacing_y),
        slice_spacing=ref_slice_spacing,
        output_dir=str(REGISTERED_INPUT_PROJECTION_OUTPUT_DIR),
        output_prefix='Animation',
        n_steps=ROTATION_STEPS,
        duration=ANIMATION_DURATION,
        contrast_window=(-20, 750)
    )

if __name__ == "__main__":
    main()


