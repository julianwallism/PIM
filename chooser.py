import numpy as np
import matplotlib.pyplot as plt
from utils import (
    rotate_on_axial_plane,
    MIP_sagittal_plane
)
from matplotlib.widgets import Slider # Added import for Slider


# Removed original script-level code from here:
# img = np.load("img.npy")
# ...
# ax.imshow(proj_img, cmap='bone', vmin=img_min, vmax=img_max, aspect=5 / 0.892578)

# Added new function for interactive viewer
def setup_interactive_mip_viewer():
    img = np.load("registered.npy")

    # Initial values for sliders, taken from the original script's context
    initial_img_min = -100
    initial_img_max = 1000

    # Prepare the first frame (angle 0)
    rotated_img = rotate_on_axial_plane(img, 0)
    proj_img = MIP_sagittal_plane(rotated_img)

    fig, ax = plt.subplots()
    # Adjust layout to make space for sliders at the bottom
    plt.subplots_adjust(left=0.1, bottom=0.30)

    # Determine overall min/max for slider range from the entire 3D image data
    data_min_val = np.min(img)
    data_max_val = np.max(img)
    
    # Ensure initial values are within the slider's actual data range and valid
    current_vmin = max(data_min_val, initial_img_min)
    current_vmax = min(data_max_val, initial_img_max)
    if current_vmin > current_vmax: # Safety check if initial values are problematic
        current_vmin = data_min_val
        current_vmax = data_max_val if data_max_val > data_min_val else data_min_val + 1


    im_display = ax.imshow(proj_img, cmap='bone', vmin=current_vmin, vmax=current_vmax, aspect=5 / 0.892578)
    ax.set_title("Interactive MIP: Adjust Min/Max Intensity (Frame 0)")

    # Slider axes ([left, bottom, width, height])
    ax_min_slider = plt.axes([0.15, 0.15, 0.7, 0.03])
    ax_max_slider = plt.axes([0.15, 0.08, 0.7, 0.03])

    # Create sliders
    # Using a small step for float data, or use valstep=1 for integer data like HU units
    slider_step = (data_max_val - data_min_val) / 500 if (data_max_val - data_min_val) > 0 else 0.1

    s_min = Slider(
        ax=ax_min_slider,
        label='Min Intensity',
        valmin=data_min_val,
        valmax=data_max_val,
        valinit=current_vmin,
        valstep=slider_step
    )

    s_max = Slider(
        ax=ax_max_slider,
        label='Max Intensity',
        valmin=data_min_val,
        valmax=data_max_val,
        valinit=current_vmax,
        valstep=slider_step
    )

    def update_min(val):
        new_min = s_min.val
        if new_min > s_max.val:
            s_max.set_val(new_min) # Ensure min <= max
        im_display.set_clim(vmin=new_min, vmax=s_max.val)
        fig.canvas.draw_idle()

    def update_max(val):
        new_max = s_max.val
        if new_max < s_min.val:
            s_min.set_val(new_max) # Ensure min <= max
        im_display.set_clim(vmin=s_min.val, vmax=new_max)
        fig.canvas.draw_idle()

    s_min.on_changed(update_min)
    s_max.on_changed(update_max)

    plt.show()

if __name__ == "__main__":
    setup_interactive_mip_viewer()
