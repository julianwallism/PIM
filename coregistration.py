import numpy as np
from scipy import ndimage
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def mse(img1, img2):
    """Calculate the mean squared error between two images"""
    return np.mean((img1.astype(np.int32) - img2.astype(np.int32)) ** 2)

def mae(img1, img2):
    """Calculate the mean absolute error between two images"""
    return np.mean(np.abs(img1.astype(np.int32) - img2.astype(np.int32)))

def mutual_information(img1, img2):
    """ Compute the Shannon Mutual Information between two images. """
    nbins = [10, 10]
    # Compute entropy of each image
    hist = np.histogram(img1.ravel(), bins=nbins[0])[0]
    prob_distr = hist / np.sum(hist)
    entropy_input = -np.sum(prob_distr * np.log2(prob_distr + 1e-7)) 
    hist = np.histogram(img2.ravel(), bins=nbins[0])[0]
    prob_distr = hist / np.sum(hist)
    entropy_reference = -np.sum(prob_distr * np.log2(prob_distr + 1e-7)) 
    
    # Compute joint entropy
    joint_hist = np.histogram2d(img1.ravel(), img2.ravel(), bins=nbins)[0]
    prob_distr = joint_hist / np.sum(joint_hist)
    joint_entropy = -np.sum(prob_distr * np.log2(prob_distr + 1e-7))
    
    # Compute mutual information
    return entropy_input + entropy_reference - joint_entropy

def calculate_centers_of_mass(img1, img2):
    """Calculate centers of mass for both images"""
    center1 = ndimage.center_of_mass(np.abs(img1))
    center2 = ndimage.center_of_mass(np.abs(img2))
    return np.array(center1), np.array(center2)


def rigid_3D_coregistration(
    reference_img: np.ndarray,
    input_img: np.ndarray,
    method: str = 'L-BFGS-B',
    maxiter: int = 100,
    maxfun: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """Perform rigid 3D registration of two images by aligning the input image to the reference image using quaternion-based transformation.
    
    Args:
        reference_img: Fixed image
        input_img: Moving image
        method: Optimization method ('L-BFGS-B', 'Powell', or 'Nelder-Mead')
        maxiter: Maximum number of iterations
        maxfun: Maximum number of function evaluations (called maxfev for Powell and Nelder-Mead)
        
    Returns:
        Registered moving image
        Transformation matrix
    """
    
    # Calculate metrics between reference and input image before registration
    mse_value = mse(input_img, reference_img)
    mae_value = mae(input_img, reference_img)
    mi_value = mutual_information(input_img, reference_img)
    print(f"Initial metrics - MSE: {mse_value:.2f}, MAE: {mae_value:.2f}, MI: {mi_value:.2f}")
    
    # Calculate centers of mass for initial translation
    com_reference, com_input = calculate_centers_of_mass(reference_img, input_img)
    
    # Initial parameters
    initial_parameters = [
        0, 0, 0,    # Translation vector
        0,          # Angle in rads
        1, 0, 0,    # Axis of rotation
    ]
    
    # Set initial translation to align centers of mass
    initial_parameters[0] = com_reference[0] - com_input[0]
    initial_parameters[1] = com_reference[1] - com_input[1]
    initial_parameters[2] = com_reference[2] - com_input[2]
    
    # Storage for loss history
    loss_history = []
    
    def function_to_minimize(parameters):
        """Calculate MSE between transformed input image and reference image"""
        t1, t2, t3, angle_in_rads, v1, v2, v3 = parameters
        v_norm = math.sqrt(sum([coord ** 2 for coord in [v1, v2, v3]]))
        v1, v2, v3 = v1/v_norm, v2/v_norm, v3/v_norm
        
        # Create rotation matrix from quaternion
        cos = math.cos(angle_in_rads / 2)
        sin = math.sin(angle_in_rads / 2)
        q = (cos, sin * v1, sin * v2, sin * v3)
        
        rotation_matrix = np.zeros((3, 3))
        rotation_matrix[0, 0] = 1 - 2 * (q[2]**2 + q[3]**2)
        rotation_matrix[0, 1] = 2 * (q[1] * q[2] - q[0] * q[3])
        rotation_matrix[0, 2] = 2 * (q[1] * q[3] + q[0] * q[2])
        rotation_matrix[1, 0] = 2 * (q[1] * q[2] + q[0] * q[3])
        rotation_matrix[1, 1] = 1 - 2 * (q[1]**2 + q[3]**2)
        rotation_matrix[1, 2] = 2 * (q[2] * q[3] - q[0] * q[1])
        rotation_matrix[2, 0] = 2 * (q[1] * q[3] - q[0] * q[2])
        rotation_matrix[2, 1] = 2 * (q[2] * q[3] + q[0] * q[1])
        rotation_matrix[2, 2] = 1 - 2 * (q[1]**2 + q[2]**2)
        
        inv_rotation = np.linalg.inv(rotation_matrix)
        offset = -np.dot(inv_rotation, [t1, t2, t3])
        
        # Apply transformation
        transformed_img = ndimage.affine_transform(
            input_img,
            matrix=inv_rotation,
            offset=offset,
            output_shape=reference_img.shape,
            order=1
        )
        
        # Calculate MSE
        current_mse = mse(transformed_img, reference_img)
        loss_history.append(current_mse)
        
        return current_mse
    
    # Run optimization with selected method
    print(f"Starting optimization with method: {method}")
    
    if method not in ['L-BFGS-B', 'Powell', 'Nelder-Mead']:
        print(f"Warning: Unknown method '{method}'. Falling back to L-BFGS-B.")
        method = 'L-BFGS-B'
    
    # Set optimization options based on the method
    if method == 'L-BFGS-B':
        options = {'disp': True, 'maxiter': maxiter, 'maxfun': maxfun}
    elif method == 'Powell':
        options = {'disp': True, 'maxiter': maxiter, 'maxfev': maxfun}
    elif method == 'Nelder-Mead':
        options = {'disp': True, 'maxiter': maxiter, 'maxfev': maxfun}
    else:
        options = {'disp': True, 'maxiter': maxiter}
    
    result = minimize(
        function_to_minimize,
        x0=initial_parameters,
        method=method,
        options=options
    )
    
    # Extract the optimized parameters
    parameters = result.x
    t1, t2, t3, angle_in_rads, v1, v2, v3 = parameters
    
    # Normalize rotation axis
    v_norm = math.sqrt(sum([coord ** 2 for coord in [v1, v2, v3]]))
    v1, v2, v3 = v1/v_norm, v2/v_norm, v3/v_norm
    
    # Create rotation matrix from quaternion
    cos = math.cos(angle_in_rads / 2)
    sin = math.sin(angle_in_rads / 2)
    q = (cos, sin * v1, sin * v2, sin * v3)
    
    # Convert quaternion to rotation matrix
    rotation_matrix = np.zeros((3, 3))
    rotation_matrix[0, 0] = 1 - 2 * (q[2]**2 + q[3]**2)
    rotation_matrix[0, 1] = 2 * (q[1] * q[2] - q[0] * q[3])
    rotation_matrix[0, 2] = 2 * (q[1] * q[3] + q[0] * q[2])
    rotation_matrix[1, 0] = 2 * (q[1] * q[2] + q[0] * q[3])
    rotation_matrix[1, 1] = 1 - 2 * (q[1]**2 + q[3]**2)
    rotation_matrix[1, 2] = 2 * (q[2] * q[3] - q[0] * q[1])
    rotation_matrix[2, 0] = 2 * (q[1] * q[3] - q[0] * q[2])
    rotation_matrix[2, 1] = 2 * (q[2] * q[3] + q[0] * q[1])
    rotation_matrix[2, 2] = 1 - 2 * (q[1]**2 + q[2]**2)
    
    # Create full transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = [t1, t2, t3]
    
    # For ndimage.affine_transform, we need the inverse mapping
    # (from output coordinates to input coordinates)
    inv_rotation = np.linalg.inv(rotation_matrix)
    offset = -np.dot(inv_rotation, [t1, t2, t3])
    
    # Apply transformation to input image
    registered_img = ndimage.affine_transform(
        input_img,
        matrix=inv_rotation,
        offset=offset,
        output_shape=reference_img.shape,
        order=1  # Linear interpolation
    )
    
    # Calculate final metrics after registration
    final_mse = mse(registered_img, reference_img)
    final_mae = mae(registered_img, reference_img)
    final_mi = mutual_information(registered_img, reference_img)
    print(f"Final metrics - MSE: {final_mse:.2f}, MAE: {final_mae:.2f}, MI: {final_mi:.2f}")
    print(f"Optimization success: {result.success}, iterations: {result.nit}, function evaluations: {result.nfev}")
    
    # Plot the loss history
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title(f'MSE Loss During Optimization ({method})')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f'registration_loss_{method}.png'
    plt.savefig(plot_filename)
    print(f"Loss plot saved as {plot_filename}")
    
    return registered_img, transformation_matrix


def compare_optimization_methods(
    reference_img, 
    input_img, 
    maxiter: int = 100, 
    maxfun: int = 200
):
    """Compare different optimization methods for image registration
    
    Args:
        reference_img: Fixed image
        input_img: Moving image
        maxiter: Maximum number of iterations
        maxfun: Maximum number of function evaluations
    
    Returns:
        Best registered image
        Best transformation matrix
    """
    methods = ['L-BFGS-B', 'Powell', 'Nelder-Mead']
    results = {}
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Running registration with {method} optimizer")
        print(f"{'='*50}")
        
        registered_img, transformation_matrix = rigid_3D_coregistration(
            reference_img=reference_img,
            input_img=input_img,
            method=method,
            maxiter=maxiter,
            maxfun=maxfun
        )
        
        results[method] = {
            'registered_img': registered_img,
            'transformation': transformation_matrix,
            'mse': mse(registered_img, reference_img),
            'mae': mae(registered_img, reference_img),
            'mi': mutual_information(registered_img, reference_img)
        }
    
    # Compare results
    print("\nComparison of optimization methods:")
    print(f"{'Method':<12} {'MSE':<10} {'MAE':<10} {'MI':<10}")
    print("-" * 42)
    
    for method, result in results.items():
        print(f"{method:<12} {result['mse']:<10.2f} {result['mae']:<10.2f} {result['mi']:<10.2f}")
    
    # Find best method
    best_method = min(results.items(), key=lambda x: x[1]['mse'])[0]
    print(f"\nBest method based on MSE: {best_method}")
    
    return results[best_method]['registered_img'], results[best_method]['transformation']

    
    
