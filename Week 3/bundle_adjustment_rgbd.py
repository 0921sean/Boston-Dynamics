import numpy as np
import cv2
from scipy.optimize import least_squares

def bundle_adjustment_rgbd(keypoints, matches_list, camera_matrix, depth_images):
    """Implement bundle adjustment for RGB-D images"""
    # Similar structure as RGB, but incorporate depth information
    point_indices = {}
    points_3d = []
    points_2d = []
    depth_values = []  # Depth values for each observation
    camera_indices = []
    point_indices_list = []
    
    global_point_idx = 0
    
    # Process each pair of consecutive images
    for img_idx, matches in enumerate(matches_list):
        for match in matches:
            idx1 = match.queryIdx
            idx2 = match.trainIdx
            
            key1 = (img_idx, idx1)
            key2 = (img_idx + 1, idx2)
            
            if key1 not in point_indices:
                point_indices[key1] = global_point_idx
                point_indices[key2] = global_point_idx
                
                # Get depth and initialize 3D point
                x1, y1 = keypoints[img_idx][idx1].pt
                depth1 = depth_images[img_idx][int(y1), int(x1)]
                
                if depth1 > 0:  # Valid depth
                    # Convert from image to camera coordinates
                    x1_norm = (x1 - camera_matrix[0, 2]) / camera_matrix[0, 0]
                    y1_norm = (y1 - camera_matrix[1, 2]) / camera_matrix[1, 1]
                    
                    # Use depth to initialize the 3D point
                    X = x1_norm * depth1
                    Y = y1_norm * depth1
                    Z = depth1
                    
                    points_3d.append([X, Y, Z])
                else:
                    # Fallback if depth is invalid
                    points_3d.append([0, 0, 0])
                
                global_point_idx += 1
            else:
                point_indices[key2] = point_indices[key1]
            
            # Add observations
            points_2d.append(keypoints[img_idx][idx1].pt)
            x1, y1 = keypoints[img_idx][idx1].pt
            depth1 = depth_images[img_idx][int(y1), int(x1)]
            depth_values.append(depth1 if depth1 > 0 else 0)
            camera_indices.append(img_idx)
            point_indices_list.append(point_indices[key1])
            
            points_2d.append(keypoints[img_idx + 1][idx2].pt)
            x2, y2 = keypoints[img_idx + 1][idx2].pt
            depth2 = depth_images[img_idx + 1][int(y2), int(x2)]
            depth_values.append(depth2 if depth2 > 0 else 0)
            camera_indices.append(img_idx + 1)
            point_indices_list.append(point_indices[key1])
    
    # Define the cost function for RGBD bundle adjustment
    def bundle_adjustment_rgbd_cost(params):
        """Cost function for RGBD bundle adjustment"""
        n_cameras = len(keypoints)
        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6:].reshape((global_point_idx, 3))
        
        errors = []
        
        for i in range(len(points_2d)):
            cam_idx = camera_indices[i]
            point_idx = point_indices_list[i]
            depth = depth_values[i]
            
            r = camera_params[cam_idx, :3]
            t = camera_params[cam_idx, 3:6]
            
            R, _ = cv2.Rodrigues(r)
            
            p = points_3d[point_idx]
            p_proj = R @ p + t
            
            # Image reprojection error
            x = p_proj[0] / p_proj[2]
            y = p_proj[1] / p_proj[2]
            
            x_proj = camera_matrix[0, 0] * x + camera_matrix[0, 2]
            y_proj = camera_matrix[1, 1] * y + camera_matrix[1, 2]
            
            x_obs, y_obs = points_2d[i]
            errors.append(x_proj - x_obs)
            errors.append(y_proj - y_obs)
            
            # Depth error (if valid depth is available)
            if depth > 0:
                errors.append(p_proj[2] - depth)
        
        return np.array(errors)
    
    # Initial camera parameters
    initial_cameras = np.zeros((len(keypoints), 6))
    for i in range(1, len(keypoints)):
        initial_cameras[i, 3] = i
    
    # Concatenate parameters
    initial_params = np.hstack((initial_cameras.ravel(), np.array(points_3d).ravel()))
    
    # Run optimization
    result = least_squares(bundle_adjustment_rgbd_cost, initial_params, method='trf', max_nfev=100)
    
    # Extract results
    n_cameras = len(keypoints)
    optimized_cameras = result.x[:n_cameras * 6].reshape((n_cameras, 6))
    optimized_points = result.x[n_cameras * 6:].reshape((global_point_idx, 3))
    
    return optimized_cameras, optimized_points