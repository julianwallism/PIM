#!/usr/bin/env python3
"""
DICOM Visualizer for CT and Segmentation Visualization
This script loads CT DICOM images and associated segmentations,
and rearranges them based on DICOM headers.
"""

import os
import sys
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from scipy import ndimage
from pathlib import Path
import logging
from skimage import measure

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_ct_series(directory):
    """Load a series of DICOM CT images from a directory."""
    logger.info(f"Loading CT series from: {directory}")
    slices = []
    
    # Load all DICOM files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.dcm'):
            file_path = os.path.join(directory, filename)
            ds = pydicom.dcmread(file_path)
            if hasattr(ds, 'Modality') and ds.Modality == 'CT':
                slices.append(ds)

    
    if not slices:
        raise ValueError(f"No valid CT DICOM files found in {directory}")
    
    logger.info(f"Found {len(slices)} CT slices")
    
    # Sort slices based on position
    def get_slice_position(ds):
        if hasattr(ds, 'ImagePositionPatient'):
            return float(ds.ImagePositionPatient[2])
        elif hasattr(ds, 'SliceLocation'):
            return float(ds.SliceLocation)
        elif hasattr(ds, 'InstanceNumber'):
            return float(ds.InstanceNumber)
        else:
            return 0.0
    
    # Sort slices and create volume
    sorted_slices = sorted(slices, key=get_slice_position)
    volume_array = np.stack([s.pixel_array for s in sorted_slices])
    
    # Convert to Hounsfield Units if available
    if hasattr(sorted_slices[0], 'RescaleSlope') and hasattr(sorted_slices[0], 'RescaleIntercept'):
        slope = sorted_slices[0].RescaleSlope
        intercept = sorted_slices[0].RescaleIntercept
        volume_array = volume_array * slope + intercept
    
    # Collect metadata
    metadata = {
        'patient_id': getattr(sorted_slices[0], 'PatientID', 'Unknown'),
        'study_date': getattr(sorted_slices[0], 'StudyDate', 'Unknown'),
        'series_description': getattr(sorted_slices[0], 'SeriesDescription', 'Unknown'),
        'z_positions': [get_slice_position(s) for s in sorted_slices],
        'slice_thickness': getattr(sorted_slices[0], 'SliceThickness', 1.0),
        'pixel_spacing': getattr(sorted_slices[0], 'PixelSpacing', [1.0, 1.0]),
        'number_of_slices': len(sorted_slices),
    }
    
    logger.info(f"Loaded volume with shape: {volume_array.shape}")
    return sorted_slices, volume_array, metadata

def load_segmentation(seg_file_path, ct_slices):
    """Load a DICOM segmentation file and align it with the CT volume."""
    logger.info(f"Loading segmentation from: {seg_file_path}")

    # Load the segmentation file
    seg_dataset = pydicom.dcmread(seg_file_path)
    
    # Check if this is a segmentation object
    if not hasattr(seg_dataset, 'SegmentSequence'):
        logger.warning(f"File {seg_file_path} does not appear to be a valid DICOM segmentation")
        return {}
    
    # Get number of segments
    num_segments = len(seg_dataset.SegmentSequence)
    logger.info(f"Found {num_segments} segments in the segmentation file")
    
    # Get CT slice positions for matching
    ct_positions = [float(ct.ImagePositionPatient[2]) if hasattr(ct, 'ImagePositionPatient') else i 
                    for i, ct in enumerate(ct_slices)]
    
    # Get segmentation pixel data
    if not hasattr(seg_dataset, 'pixel_array'):
        logger.error("No pixel data found in segmentation file")
        return {}
        
    seg_pixel_data = seg_dataset.pixel_array
    segment_masks = {}
    
    # Process each segment
    for segment_number in range(1, num_segments + 1):
        # Create empty mask matching the CT volume
        mask = np.zeros((len(ct_slices), ct_slices[0].pixel_array.shape[0], ct_slices[0].pixel_array.shape[1]), 
                        dtype=bool)
        
        # Find frames for this segment
        segment_frames = []
        
        # Extract frames for current segment
        if hasattr(seg_dataset, 'PerFrameFunctionalGroupsSequence'):
            for frame_idx, segment_idx in enumerate(seg_dataset.PerFrameFunctionalGroupsSequence):
                if hasattr(segment_idx, 'SegmentIdentificationSequence'):
                    segment_id = segment_idx.SegmentIdentificationSequence[0]
                    if hasattr(segment_id, 'ReferencedSegmentNumber') and segment_id.ReferencedSegmentNumber == segment_number:
                        segment_frames.append(frame_idx)
        
        # Map segmentation frames to CT slices
        for frame_idx in segment_frames:
            if hasattr(seg_dataset, 'PerFrameFunctionalGroupsSequence'):
                frame_item = seg_dataset.PerFrameFunctionalGroupsSequence[frame_idx]
                if hasattr(frame_item, 'PlanePositionSequence') and hasattr(frame_item.PlanePositionSequence[0], 'ImagePositionPatient'):
                    frame_z = float(frame_item.PlanePositionSequence[0].ImagePositionPatient[2])
                    
                    # Find the closest CT slice
                    ct_idx = np.argmin(np.abs(np.array(ct_positions) - frame_z))
                    
                    # Extract the binary mask for this frame
                    frame_data = seg_pixel_data[frame_idx]
                    
                    # Add to the 3D mask
                    mask[ct_idx] = (frame_data > 0)
        
        # Get segment name
        segment_name = f"Segment {segment_number}"
        if hasattr(seg_dataset, 'SegmentSequence'):
            segment_seq = seg_dataset.SegmentSequence[segment_number-1]
            if hasattr(segment_seq, 'SegmentLabel'):
                segment_name = segment_seq.SegmentLabel
        
        # Store the mask
        segment_masks[segment_number] = {
            'mask': mask,
            'name': segment_name
        }
        
        logger.info(f"Processed segment {segment_number}: {segment_name}")
    
    return segment_masks

def get_contour_mask(binary_mask):
    """Find contours in a binary mask using erosion."""
    eroded = ndimage.binary_erosion(binary_mask)
    return np.logical_and(binary_mask, np.logical_not(eroded))

def save_ct_and_segmentation_as_gif(ct_volume, segmentations, metadata, output_path='ct_segmentation.gif', fps=10):
    """Save CT volume with overlaid segmentations as an animated GIF."""
    logger.info(f"Creating GIF animation at {fps} fps")
    
    # Window the CT data for better visualization (default soft tissue window)
    window_center, window_width = 40, 400
    ct_display = np.clip(ct_volume, window_center - window_width/2, window_center + window_width/2)
    ct_display = (ct_display - (window_center - window_width/2)) / window_width * 255
    
    # Get dimensions and spacing information
    n_slices, height, width = ct_volume.shape
    pixel_spacing = metadata.get('pixel_spacing', [1.0, 1.0])
    slice_thickness = metadata.get('slice_thickness', 1.0)
    z_positions = metadata.get('z_positions', None)
    
    # Calculate aspect ratios
    slice_aspect_x = slice_thickness / pixel_spacing[0]
    slice_aspect_y = slice_thickness / pixel_spacing[1]
    
    # Setup the figure
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    ax_coronal = plt.subplot(gs[0])
    ax_sagittal = plt.subplot(gs[1])
    
    # Style information
    contour_colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF']
    line_widths = [2, 1, 1, 1, 1, 1]
    
    # Create legend elements
    legend_elements = []
    if segmentations:
        for i, (seg_num, seg_data) in enumerate(segmentations.items()):
            color = contour_colors[i % len(contour_colors)]
            from matplotlib.lines import Line2D
            legend_elements.append(Line2D([0], [0], color=color, 
                                         linewidth=line_widths[i % len(line_widths)],
                                         label=seg_data['name']))
    
    # Set title
    title_str = f"Patient: {metadata['patient_id']} - {metadata['series_description']}"
    fig.suptitle(title_str, fontsize=14)
    
    # Total number of frames
    total_frames = max(height, width)
    
    # Function to update the figure for each frame
    def update_frame(frame_num):
        # Calculate positions for each plane
        coronal_idx = min(frame_num * height // total_frames, height - 1)
        sagittal_idx = min(frame_num * width // total_frames, width - 1)
        
        # Clear axes
        ax_coronal.clear()
        ax_sagittal.clear()
        
        # Add position information if available
        coronal_mm = f" (Position: {coronal_idx * pixel_spacing[1]:.1f} mm)" if z_positions else ""
        sagittal_mm = f" (Position: {sagittal_idx * pixel_spacing[0]:.1f} mm)" if z_positions else ""
        
        # Set titles
        ax_coronal.set_title(f'Coronal Plane{coronal_mm}')
        ax_sagittal.set_title(f'Sagittal Plane{sagittal_mm}')
        
        # Display CT images
        coronal_slice = np.flipud(ct_display[:, coronal_idx, :])  # Flip for correct orientation
        sagittal_slice = np.flipud(ct_display[:, :, sagittal_idx])
        
        ax_coronal.imshow(coronal_slice, cmap='gray', aspect=slice_aspect_x)
        ax_sagittal.imshow(sagittal_slice, cmap='gray', aspect=slice_aspect_y)
        
        # Add segmentation contours
        for i, (seg_num, seg_data) in enumerate(segmentations.items()):
            color = contour_colors[i % len(contour_colors)]
            line_width = line_widths[i % len(line_widths)]
            seg_mask = seg_data['mask']
            
            # Process coronal plane
            if coronal_idx < seg_mask.shape[1]:
                # Extract and flip mask to match orientation
                coronal_mask = np.flipud(seg_mask[:, coronal_idx, :])
                contours = measure.find_contours(coronal_mask.astype(float), 0.5)
                for contour in contours:
                    ax_coronal.plot(contour[:, 1], contour[:, 0], color=color, linewidth=line_width)
            
            # Process sagittal plane
            if sagittal_idx < seg_mask.shape[2]:
                sagittal_mask = np.flipud(seg_mask[:, :, sagittal_idx])
                contours = measure.find_contours(sagittal_mask.astype(float), 0.5)
                for contour in contours:
                    ax_sagittal.plot(contour[:, 1], contour[:, 0], color=color, linewidth=line_width)
        
        # Show Z-range information
        if z_positions is not None:
            z_min, z_max = min(z_positions), max(z_positions)
            ax_coronal.set_xlabel(f"Slice: {coronal_idx+1}/{height} (Z-range: {z_min:.1f}-{z_max:.1f} mm)")
            ax_sagittal.set_xlabel(f"Slice: {sagittal_idx+1}/{width} (Z-range: {z_min:.1f}-{z_max:.1f} mm)")
        else:
            ax_coronal.set_xlabel(f"Slice: {coronal_idx+1}/{height}")
            ax_sagittal.set_xlabel(f"Slice: {sagittal_idx+1}/{width}")
        
        # Add legend on first frame
        if segmentations and frame_num == 0:
            fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.95, 0.95))
        
        return [ax_coronal, ax_sagittal]
    
    # Create and save animation
    logger.info("Generating animation frames...")
    anim = animation.FuncAnimation(
        fig, update_frame, frames=total_frames,
        interval=1000/fps, blit=False
    )
    
    logger.info(f"Saving animation to {output_path}")
    anim.save(
        output_path, writer='pillow', fps=fps, dpi=120
    )
    
    plt.close(fig)
    logger.info(f"GIF saved successfully to {output_path}")

def main():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RadCTTACEomics_2230/2230")
    ct_dir = os.path.join(base_dir, "21_PP_Ax5.00mm")
    seg_files = [
        os.path.join(base_dir, "21_PP_Ax5.00mm_ManualROI_Liver.dcm"),
        os.path.join(base_dir, "21_PP_Ax5.00mm_ManualROI_Tumor.dcm")
    ]
    output_path = 'ct_segmentation.gif'
    fps = 10

    # Load CT series
    ct_slices, ct_volume, metadata = load_ct_series(ct_dir)
    
    # Load segmentation files
    all_segmentations = {}
    for seg_file in seg_files:
        seg_masks = load_segmentation(seg_file, ct_slices)
        all_segmentations.update(seg_masks)
    
    # Save as GIF
    save_ct_and_segmentation_as_gif(ct_volume, all_segmentations, metadata, output_path, fps)
    logger.info(f"GIF animation saved to: {output_path} at {fps} fps")

if __name__ == "__main__":
    main()