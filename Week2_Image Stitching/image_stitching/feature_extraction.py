import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_and_display_features(image_path, method="SIFT"):
    """
    Extract and display image features using specified method
    
    Parameters:
    image_path (str): Path to input image
    method (str): Feature extraction method ('SIFT', 'ORB', or 'FAST')
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return None, None
    
    # Convert to grayscale (required for feature detection)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Initialize feature detector based on method
    if method == "SIFT":
        detector = cv2.SIFT_create()
        color = (0, 255, 0)  # Green
    elif method == "ORB":
        detector = cv2.ORB_create()
        color = (255, 0, 0)  # Blue
    elif method == "FAST":
        # FAST only detects keypoints, doesn't compute descriptors
        fast = cv2.FastFeatureDetector_create()
        keypoints = fast.detect(gray, None)
        # For visualization
        img_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 0, 255))
        
        print(f"Number of {method} keypoints detected: {len(keypoints)}")
        return keypoints, None
    else:
        print(f"Error: Unknown method {method}")
        return None, None
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    
    # Draw keypoints on image
    img_keypoints = cv2.drawKeypoints(img, keypoints, None, color=color, 
                                     flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    
    print(f"Number of {method} keypoints detected: {len(keypoints)}")
    if descriptors is not None:
        print(f"Descriptor shape: {descriptors.shape}")
    
    # Display image with keypoints
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img_keypoints, cv2.COLOR_BGR2RGB))
    plt.title(f"{method} Features")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{method}_features.jpg")
    plt.show()
    
    return keypoints, descriptors

# Example usage
if __name__ == "__main__":
    image_path = "input_image.jpg"  # Replace with your image path
    
    # Extract features using different methods
    sift_kp, sift_desc = extract_and_display_features(image_path, "SIFT")
    orb_kp, orb_desc = extract_and_display_features(image_path, "ORB")
    fast_kp, fast_desc = extract_and_display_features(image_path, "FAST")