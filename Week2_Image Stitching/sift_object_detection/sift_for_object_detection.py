import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_object_with_sift(object_image_path, scene_image_path, min_match_count=10):
    """
    Detect an object in a scene using SIFT features and homography
    
    Parameters:
    object_image_path (str): Path to the object image
    scene_image_path (str): Path to the scene image containing the object
    min_match_count (int): Minimum number of good matches required
    
    Returns:
    numpy.ndarray: Result image with object highlighted
    """
    # Read images
    img_object = cv2.imread(object_image_path, cv2.IMREAD_COLOR)
    img_scene = cv2.imread(scene_image_path, cv2.IMREAD_COLOR)
    
    if img_object is None or img_scene is None:
        print("Error: Could not read one or both images")
        return None
    
    # Convert to grayscale
    gray_object = cv2.cvtColor(img_object, cv2.COLOR_BGR2GRAY)
    gray_scene = cv2.cvtColor(img_scene, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Find keypoints and descriptors
    kp_object, des_object = sift.detectAndCompute(gray_object, None)
    kp_scene, des_scene = sift.detectAndCompute(gray_scene, None)
    
    # Check if descriptors were found
    if des_object is None or des_scene is None:
        print("Error: No descriptors found in one or both images")
        return None
    
    # FLANN parameters for matching
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    # FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Match descriptors
    matches = flann.knnMatch(des_object, des_scene, k=2)
    
    # Apply ratio test to find good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    print(f"Total SIFT keypoints in object: {len(kp_object)}")
    print(f"Total SIFT keypoints in scene: {len(kp_scene)}")
    print(f"Number of good matches: {len(good_matches)}")
    
    # Create a copy of the scene image for visualization
    img_result = img_scene.copy()
    
    # If we have enough good matches, find the object
    if len(good_matches) >= min_match_count:
        # Extract locations of matched keypoints
        src_pts = np.float32([kp_object[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography using RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        
        # Get the corners of the object
        h, w = gray_object.shape
        corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        
        # Transform object corners to scene image
        scene_corners = cv2.perspectiveTransform(corners, H)
        
        # Draw the object boundary in the scene
        cv2.polylines(img_result, [np.int32(scene_corners)], True, (0, 255, 0), 3)
        
        # Count inliers (matches that passed RANSAC)
        inlier_count = np.sum(mask)
        print(f"Number of inliers: {inlier_count}")
        
        # Draw inlier matches
        draw_params = dict(
            matchColor=(0, 255, 0),  # Green color for matches
            singlePointColor=None,
            matchesMask=matchesMask,  # Only draw inliers
            flags=2
        )
        
        img_matches = cv2.drawMatches(img_object, kp_object, img_scene, kp_scene, good_matches, None, **draw_params)
        
        return img_result, img_matches
    else:
        print(f"Not enough good matches: {len(good_matches)}/{min_match_count}")
        return img_scene, None

def main():
    # Replace with your image paths
    object_path = "alexander_book.jpeg"  # Image of the object you want to detect
    scene_path = "alexander_book_degraded.jpeg"    # Image of the scene where the object might be present
    
    # Detect object in scene
    result_img, matches_img = detect_object_with_sift(object_path, scene_path)
    
    if result_img is not None:
        # Display results
        plt.figure(figsize=(12, 6))
        
        if matches_img is not None:
            plt.subplot(121)
            plt.imshow(cv2.cvtColor(matches_img, cv2.COLOR_BGR2RGB))
            plt.title("SIFT Matches (Inliers)")
            plt.axis('off')
            
            plt.subplot(122)
        
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title("Object Detection Result")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("find_alexander_book_degraded.jpg", dpi=300)
        plt.show()

if __name__ == "__main__":
    main()