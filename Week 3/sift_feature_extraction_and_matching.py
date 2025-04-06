import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def extract_and_match_features(images):
    """Extract SIFT features and match them across image pairs"""
    sift = cv2.SIFT_create()
    
    # Store keypoints and descriptors for each image
    all_keypoints = []
    all_descriptors = []
    
    # Extract features from each image
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        all_keypoints.append(keypoints)
        all_descriptors.append(descriptors)
    
    # Match features between consecutive image pairs
    matches_list = []
    for i in range(len(images)-1):
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(all_descriptors[i], all_descriptors[i+1], k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        matches_list.append(good_matches)
    
    return all_keypoints, all_descriptors, matches_list