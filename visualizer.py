"""Visualization utilities for medical images."""

from typing import Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
import os
from tqdm import tqdm
from utils import (
    MIP_sagittal_plane,
    rotate_on_axial_plane
)

def create_animation(img: np.ndarray,
                    masks: Dict[str, np.ndarray],
                    colormaps: Dict[str, ListedColormap],
                    pixel_spacing: Tuple[float, float],
                    slice_spacing: float,
                    output_dir: str,
                    output_prefix: str,
                    n_steps: int = 16,
                    duration: int = 250,
                    contrast_window: Tuple[int, int] = (-100, 1000)) -> None:
    """Create rotation animation with multiple mask overlays.
    
    Args:
        img: 3D image array
        masks: Dictionary of mask name to 3D mask array
        colormaps: Dictionary of mask name to colormap
        pixel_spacing: Tuple of (x, y) pixel spacing
        slice_spacing: Slice spacing
        output_dir: Directory to save frames
        output_prefix: Prefix for output filenames
        n_steps: Number of rotation steps
        duration: Frame duration in milliseconds
        contrast_window: Tuple of (min, max) intensity values for contrast window
    """
    img_min, img_max = contrast_window
    for idx, alpha in tqdm(enumerate(np.linspace(0, 360*(n_steps-1)/n_steps, num=n_steps)), 
                          total=n_steps, 
                          desc=f"Creating {output_prefix} frames"):
        # Rotate image and masks
        rotated_img = rotate_on_axial_plane(img, alpha)
        rotated_masks = {
            name: rotate_on_axial_plane(mask, alpha)
            for name, mask in masks.items()
        }
        
        # Create MIP projections
        proj_img = MIP_sagittal_plane(rotated_img)
        proj_masks = {
            name: MIP_sagittal_plane(mask)
            for name, mask in rotated_masks.items()
        }
        
        # Plot and save frame
        fig, ax = plt.subplots()
        ax.imshow(proj_img, cmap='bone', vmin=img_min, vmax=img_max,
                 aspect=slice_spacing / pixel_spacing[0])
        
        # Add mask overlays
        for name, proj_mask in proj_masks.items():
            ax.imshow(proj_mask > 0, cmap=colormaps[name], interpolation='none',
                     aspect=slice_spacing / pixel_spacing[0])
        
        ax.axis('off')
        
        frame_path = os.path.join(output_dir, f'{output_prefix}_{idx}.png')
        fig.savefig(frame_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    
    # Create GIF
    image_files = sorted([f for f in os.listdir(output_dir) if f.startswith(output_prefix) and f.endswith('.png')],
                        key=lambda x: int(x.split('_')[-1].split('.')[0]))
    images = [Image.open(os.path.join(output_dir, f)) for f in image_files]
    
    gif_path = os.path.join(output_dir, f'{output_prefix}.gif')
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )