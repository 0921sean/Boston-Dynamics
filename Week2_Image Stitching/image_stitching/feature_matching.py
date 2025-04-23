import cv2
import numpy as np
import matplotlib.pyplot as plt

def stitch_images(img1_path, img2_path, method="SIFT"):
    """
    Stitch two images together using feature matching
    
    Parameters:
    img1_path (str): Path to first image
    img2_path (str): Path to second image
    method (str): Feature extraction method ('SIFT', 'ORB')
    """
    # Read images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("Error: Could not read one or both images")
        return None
        
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Initialize feature detector and descriptor
    if method == "SIFT":
        detector = cv2.SIFT_create()
    elif method == "ORB":
        detector = cv2.ORB_create(nfeatures=1500)
    else:
        print(f"Error: Method {method} not supported for stitching")
        return None
    
    # Find keypoints and descriptors
    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)
    
    print(f"Image 1: {len(kp1)} keypoints")
    print(f"Image 2: {len(kp2)} keypoints")
    
    # Match features between the images
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
    
    print(f"Number of good matches: {len(good_matches)}")
    
    # Draw matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, 
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Display matches
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(f"Feature Matching using {method}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{method}_matches.jpg")
    plt.show()
    
    # If we have enough matches, find homography and warp images
    if len(good_matches) >= 4:
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
        
        # Display result
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title("Stitched Image")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("stitched_image.jpg")
        plt.show()
        
        return result
    else:
        print("Not enough good matches to stitch images")
        return None

# Example usage
if __name__ == "__main__":
    img1_path = "left_image.jpg"  # Replace with your left image path
    img2_path = "right_image.jpg"  # Replace with your right image path
    
    # Stitch using SIFT (generally better quality)
    stitched_img = stitch_images(img1_path, img2_path, method="SIFT")
    
    # You can also try ORB if SIFT is too slow
    # stitched_img = stitch_images(img1_path, img2_path, method="ORB")