import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from bundle_adjustment_rgb import bundle_adjustment_rgb
from bundle_adjustment_rgbd import bundle_adjustment_rgbd
from camera_calibration import calibrate_camera
from sift_feature_extraction_and_matching import extract_and_match_features

def main():
    # Load RGB images
    rgb_images = [cv2.imread(f'image_{i}.jpg') for i in range(5)]  # Adjust filenames as needed
    
    # For RGB-D, load both RGB and depth images
    depth_images = [cv2.imread(f'depth_{i}.png', cv2.IMREAD_ANYDEPTH) for i in range(5)]
    
    # 1. Camera calibration
    camera_matrix, dist_coeffs = calibrate_camera(rgb_images)
    print("Camera Matrix:", camera_matrix)
    
    # 2. Feature extraction and matching
    keypoints, descriptors, matches_list = extract_and_match_features(rgb_images)
    
    # 3. Bundle adjustment for RGB
    print("Running RGB bundle adjustment...")
    cameras_rgb, points_3d_rgb = bundle_adjustment_rgb(keypoints, matches_list, camera_matrix)
    
    # 4. Bundle adjustment for RGB-D
    print("Running RGB-D bundle adjustment...")
    cameras_rgbd, points_3d_rgbd = bundle_adjustment_rgbd(
        keypoints, matches_list, camera_matrix, depth_images
    )
    
    # 5. Visualize results
    visualize_reconstruction(rgb_images, cameras_rgb, points_3d_rgb, "RGB Reconstruction")
    visualize_reconstruction(rgb_images, cameras_rgbd, points_3d_rgbd, "RGB-D Reconstruction")
    
    # Compare results
    compare_reconstructions(cameras_rgb, points_3d_rgb, cameras_rgbd, points_3d_rgbd)

def visualize_reconstruction(images, cameras, points_3d, title):
    """Visualize the 3D reconstruction"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot 3D points
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='blue', marker='o', s=1)
    
    # Plot camera positions
    for i, cam in enumerate(cameras):
        r = cam[:3]
        t = cam[3:6]
        
        R, _ = cv2.Rodrigues(r)
        
        # Camera center in world coordinates: C = -R^T * t
        C = -R.T @ t
        
        ax.scatter(C[0], C[1], C[2], c='red', marker='^', s=50)
        ax.text(C[0], C[1], C[2], f'Cam {i}')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}.png')
    plt.show()

def compare_reconstructions(cameras_rgb, points_3d_rgb, cameras_rgbd, points_3d_rgbd):
    """Compare RGB and RGB-D reconstructions"""
    # Compute scale difference (assuming RGB-D is metric)
    # This is a simplistic approach - you might need more sophisticated alignment
    
    # Calculate mean distance from first camera in each reconstruction
    mean_dist_rgb = np.mean(np.linalg.norm(points_3d_rgb, axis=1))
    mean_dist_rgbd = np.mean(np.linalg.norm(points_3d_rgbd, axis=1))
    
    scale_factor = mean_dist_rgbd / mean_dist_rgb if mean_dist_rgb > 0 else 1.0
    
    # Scale RGB reconstruction to match RGB-D scale
    points_3d_rgb_scaled = points_3d_rgb * scale_factor
    
    # Compute average point distance after scaling
    if len(points_3d_rgb) == len(points_3d_rgbd):
        point_distances = np.linalg.norm(points_3d_rgb_scaled - points_3d_rgbd, axis=1)
        avg_distance = np.mean(point_distances)
        
        print(f"Average point distance between reconstructions: {avg_distance:.4f} units")
        print(f"Scale factor applied to RGB: {scale_factor:.4f}")
    else:
        print("Point counts differ between reconstructions, cannot directly compare")
    
    # Compare camera positions
    print("\nCamera Position Comparison:")
    for i in range(len(cameras_rgb)):
        r_rgb = cameras_rgb[i, :3]
        t_rgb = cameras_rgb[i, 3:6]
        
        r_rgbd = cameras_rgbd[i, :3]
        t_rgbd = cameras_rgbd[i, 3:6]
        
        R_rgb, _ = cv2.Rodrigues(r_rgb)
        R_rgbd, _ = cv2.Rodrigues(r_rgbd)
        
        C_rgb = -R_rgb.T @ t_rgb
        C_rgbd = -R_rgbd.T @ t_rgbd
        
        C_rgb_scaled = C_rgb * scale_factor
        
        dist = np.linalg.norm(C_rgb_scaled - C_rgbd)
        print(f"Camera {i} position difference: {dist:.4f} units")

if __name__ == "__main__":
    main()