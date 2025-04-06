from sift_feature_extraction_and_matching import extract_and_match_features
from camera_calibration import calibrate_camera
from bundle_adjustment_rgb import bundle_adjustment_rgb
from bundle_adjustment_rgbd import bundle_adjustment_rgbd
from visualization import visualize_reconstruction, compare_reconstructions
import numpy as np
import cv2

def create_virtual_scene():
    """Create a virtual scene with known 3D points and camera parameters"""
    # Create a grid of 3D points
    x, y = np.meshgrid(range(-5, 6, 2), range(-5, 6, 2))
    z = np.ones_like(x) * 10  # Points on a plane
    
    points_3d = np.column_stack((x.flatten(), y.flatten(), z.flatten())).astype(np.float64)
    points_3d += np.random.normal(0, 0.1, points_3d.shape)
    
    # Create synthetic cameras
    n_cameras = 5
    cameras = []
    images = []
    depth_images = []
    
    # Camera intrinsics
    K = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ])
    
    # Move cameras in a circular path
    for i in range(n_cameras):
        # Camera extrinsics
        angle = i * 2 * np.pi / n_cameras
        camera_pos = np.array([5*np.cos(angle), 5*np.sin(angle), 0])
        
        # Look at the center of the point cloud
        center = np.mean(points_3d, axis=0)
        direction = center - camera_pos
        direction = direction / np.linalg.norm(direction)
        
        # Create a rotation matrix
        z_axis = direction
        x_axis = np.array([0, 0, 1])
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        
        R = np.column_stack((x_axis, y_axis, z_axis))
        t = -R @ camera_pos
        
        # Camera parameters as rotation vector and translation
        r, _ = cv2.Rodrigues(R)
        cameras.append(np.concatenate((r.flatten(), t.flatten())))
        
        # Project 3D points to image
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.zeros((480, 640), dtype=np.float32)
        
        for point in points_3d:
            # Project point to image
            p_cam = R @ point + t
            if p_cam[2] <= 0:  # Behind camera
                continue
                
            p_img = K @ (p_cam / p_cam[2])
            x, y = int(p_img[0]), int(p_img[1])
            
            if 0 <= x < 640 and 0 <= y < 480:
                image[y, x] = [255, 255, 255]  # White point
                depth[y, x] = p_cam[2]  # Store depth
        
        # Add some gaussian blur to make it more realistic
        image = cv2.GaussianBlur(image, (3, 3), 0)
        
        images.append(image)
        depth_images.append(depth)
    
    return images, depth_images, K, cameras, points_3d

def main():
    # 가상 이미지 생성
    virtual_images, virtual_depths, virtual_K, true_cameras, true_points = create_virtual_scene()
    
    # 특징점 추출 및 매칭 (sift_feature_extraction_and_matching.py에서 가져온 함수)
    keypoints, descriptors, matches_list = extract_and_match_features(virtual_images)
    
    # RGB 번들 조정 (bundle_adjustment_rgb.py에서 가져온 함수)
    cameras_rgb, points_3d_rgb = bundle_adjustment_rgb(keypoints, matches_list, virtual_K)
    
    # RGB-D 번들 조정 (bundle_adjustment_rgbd.py에서 가져온 함수)
    cameras_rgbd, points_3d_rgbd = bundle_adjustment_rgbd(
        keypoints, matches_list, virtual_K, virtual_depths
    )
    
    # 결과 시각화 (visualization.py에서 가져온 함수)
    visualize_reconstruction(virtual_images, cameras_rgb, points_3d_rgb, "RGB 재구성")
    visualize_reconstruction(virtual_images, cameras_rgbd, points_3d_rgbd, "RGB-D 재구성")
    
    # 결과 비교 (visualization.py에서 가져온 함수)
    compare_reconstructions(cameras_rgb, points_3d_rgb, cameras_rgbd, points_3d_rgbd)

if __name__ == "__main__":
    main()