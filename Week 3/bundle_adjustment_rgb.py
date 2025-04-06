import numpy as np
from scipy.optimize import least_squares
import cv2

def bundle_adjustment_rgb(keypoints, matches_list, camera_matrix):
    """Implement bundle adjustment for RGB images"""
    # Extract matched points
    point_indices = {}  # Maps (image_idx, keypoint_idx) to global point idx
    points_3d = []      # List of 3D points
    points_2d = []      # List of 2D observations
    camera_indices = [] # Camera index for each observation
    point_indices_list = [] # Point index for each observation
    
    global_point_idx = 0
    
    # Process each pair of consecutive images
    for img_idx, matches in enumerate(matches_list):
        for match in matches:
            # Get indices of matching keypoints
            idx1 = match.queryIdx
            idx2 = match.trainIdx
            
            # Track point across multiple images
            key1 = (img_idx, idx1)
            key2 = (img_idx + 1, idx2)
            
            if key1 not in point_indices:
                point_indices[key1] = global_point_idx
                point_indices[key2] = global_point_idx
                
                # Initialize 3D point (will be optimized)
                points_3d.append([0, 0, 0])  # Placeholder values
                global_point_idx += 1
            else:
                point_indices[key2] = point_indices[key1]
            
            # Add observations
            points_2d.append(keypoints[img_idx][idx1].pt)
            camera_indices.append(img_idx)
            point_indices_list.append(point_indices[key1])
            
            points_2d.append(keypoints[img_idx + 1][idx2].pt)
            camera_indices.append(img_idx + 1)
            point_indices_list.append(point_indices[key1])
    
    # Define the parameter vector for optimization
    # [camera params (rotation, translation), 3D point coordinates]
    
    # Define the cost function for bundle adjustment
    def bundle_adjustment_cost(params):
        """Cost function for bundle adjustment"""
        # Extract camera parameters and 3D points from params
        n_cameras = len(keypoints)
        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6:].reshape((global_point_idx, 3))
        
        errors = []
        
        # Project each 3D point to each camera and compute reprojection error
        for i in range(len(points_2d)):
            cam_idx = camera_indices[i]
            point_idx = point_indices_list[i]
            
            # Extract camera parameters
            r = camera_params[cam_idx, :3]
            t = camera_params[cam_idx, 3:6]
            
            # Create rotation matrix from rodriguez rotation vector
            R, _ = cv2.Rodrigues(r)
            
            # Project 3D point to image plane
            p = points_3d[point_idx]
            p_proj = R @ p + t
            
            # Convert to 2D
            x = p_proj[0] / p_proj[2]
            y = p_proj[1] / p_proj[2]
            
            # Apply camera matrix
            x_proj = camera_matrix[0, 0] * x + camera_matrix[0, 2]
            y_proj = camera_matrix[1, 1] * y + camera_matrix[1, 2]
            
            # Compute reprojection error
            x_obs, y_obs = points_2d[i]
            errors.append(x_proj - x_obs)
            errors.append(y_proj - y_obs)
        
        return np.array(errors)
    
    # Initial camera parameters (rotation vectors and translation vectors)
    initial_cameras = np.zeros((len(keypoints), 6))
    # For simplicity, assume the first camera is at origin
    for i in range(1, len(keypoints)):
        # Add some initial displacement between cameras
        initial_cameras[i, 3] = i  # x-translation
    
    # Concatenate camera parameters and 3D points for optimization
    initial_params = np.hstack((initial_cameras.ravel(), np.array(points_3d).ravel()))
    
    # Run optimization
    result = least_squares(bundle_adjustment_cost, initial_params, method='trf', max_nfev=100)
    
    # Extract optimized results
    n_cameras = len(keypoints)
    optimized_cameras = result.x[:n_cameras * 6].reshape((n_cameras, 6))
    optimized_points = result.x[n_cameras * 6:].reshape((global_point_idx, 3))
    
    return optimized_cameras, optimized_points