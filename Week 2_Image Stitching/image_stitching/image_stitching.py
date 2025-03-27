import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def extract_features(image_path, method="SIFT"):
    """
    Extract features from an image using specified method
    
    Parameters:
    image_path (str): Path to input image
    method (str): Feature extraction method ('SIFT', 'ORB', or 'FAST')
    
    Returns:
    tuple: (keypoints, descriptors, image with keypoints)
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return None, None, None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Initialize feature detector based on method
    if method == "SIFT":
        detector = cv2.SIFT_create()
        color = (0, 255, 0)  # Green
    elif method == "ORB":
        detector = cv2.ORB_create(nfeatures=1500)
        color = (255, 0, 0)  # Blue
    elif method == "FAST":
        # FAST only detects keypoints, doesn't compute descriptors
        fast = cv2.FastFeatureDetector_create()
        keypoints = fast.detect(gray, None)
        img_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 0, 255))
        return keypoints, None, img_keypoints
    else:
        print(f"Error: Unknown method {method}")
        return None, None, None
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    
    # Draw keypoints on image
    img_keypoints = cv2.drawKeypoints(img, keypoints, None, color=color, 
                                      flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    
    return keypoints, descriptors, img_keypoints

def match_features(img1, kp1, des1, img2, kp2, des2, method="SIFT"):
    """
    Match features between two images
    
    Parameters:
    img1, img2: Input images
    kp1, kp2: Keypoints from images
    des1, des2: Descriptors from images
    method (str): Feature extraction method used
    
    Returns:
    tuple: (matched image, good matches)
    """
    if des1 is None or des2 is None:
        print("Error: No descriptors to match")
        return None, []
        
    # Match features based on method
    if method == "SIFT":
        # For SIFT, use FLANN based matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Apply ratio test to find good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    else:
        # For ORB, use Brute Force matcher with Hamming distance
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:50]  # Take top 50 matches
    
    # Draw matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, 
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return img_matches, good_matches

def stitch_images(img1, kp1, img2, kp2, good_matches):
    """
    Stitch two images together using matched features
    
    Parameters:
    img1, img2: Input images
    kp1, kp2: Keypoints from images
    good_matches: Good matches between keypoints
    
    Returns:
    numpy.ndarray: Stitched image
    """
    if len(good_matches) < 4:
        print("Not enough good matches to stitch images")
        return None
        
    # Extract location of good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Find homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # Get dimensions of images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Calculate dimensions of stitched image
    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    
    # Transform corners of img1 to img2's coordinate system
    pts1_ = cv2.perspectiveTransform(pts1, H)
    pts = np.concatenate((pts2, pts1_), axis=0)
    
    # Find min and max x, y coordinates
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    
    # Translation matrix to move to positive coordinates
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
    
    # Warp img1 using homography
    result = cv2.warpPerspective(img1, Ht.dot(H), (xmax-xmin, ymax-ymin))
    
    # Copy img2 to result at correct location
    result[t[1]:h2+t[1], t[0]:w2+t[0]] = img2
    
    return result

def main():
    # Create output directory
    output_dir = "hard_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Part 1: Feature extraction
    img1_path = "my_image1.jpeg"
    img2_path = "my_image3.jpeg"
    
    # Ensure the images exist
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print("Error: One or both input images do not exist.")
        return
    
    # Extract features using different methods
    for method in ["SIFT", "ORB", "FAST"]:
        print(f"\nExtracting features using {method}...")
        
        # Extract features from both images
        kp1, des1, img1_kp = extract_features(img1_path, method)
        kp2, des2, img2_kp = extract_features(img2_path, method)
        
        if img1_kp is None or img2_kp is None:
            print(f"Error extracting features with {method}")
            continue
            
        # Save images with keypoints
        cv2.imwrite(f"{output_dir}/{method}_keypoints_img1.jpg", img1_kp)
        cv2.imwrite(f"{output_dir}/{method}_keypoints_img2.jpg", img2_kp)
        
        print(f"Image 1: {len(kp1)} keypoints detected")
        print(f"Image 2: {len(kp2)} keypoints detected")
        
        # Skip feature matching for FAST (no descriptors)
        if method == "FAST":
            continue
            
        # Part 2: Feature matching and image stitching
        print(f"\nMatching features and stitching images using {method}...")
        
        # Read original images again
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        # Match features
        img_matches, good_matches = match_features(img1, kp1, des1, img2, kp2, des2, method)
        
        if img_matches is None:
            print(f"Error matching features with {method}")
            continue
            
        # Save matched features image
        cv2.imwrite(f"{output_dir}/{method}_matches.jpg", img_matches)
        print(f"Number of good matches: {len(good_matches)}")
        
        # Stitch images
        result = stitch_images(img1, kp1, img2, kp2, good_matches)
        
        if result is not None:
            # Save stitched image
            cv2.imwrite(f"{output_dir}/{method}_stitched.jpg", result)
            print(f"Images stitched successfully using {method}")
        else:
            print(f"Failed to stitch images using {method}")

if __name__ == "__main__":
    main()