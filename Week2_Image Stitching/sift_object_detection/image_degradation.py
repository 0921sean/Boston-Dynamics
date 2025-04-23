import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def degrade_image(image_path, blur_level=15, brightness_factor=0.4, save_path=None):
    """
    Degrade image quality by adding blur and reducing brightness
    
    Parameters:
    image_path (str): Path to the input image
    blur_level (int): Gaussian blur kernel size (odd number)
    brightness_factor (float): Factor to reduce brightness (0-1)
    save_path (str): Path to save the degraded image
    
    Returns:
    numpy.ndarray: Degraded image
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return None
    
    # Apply Gaussian blur to make the image out of focus
    # Ensure blur_level is odd
    if blur_level % 2 == 0:
        blur_level += 1
    
    blurred_img = cv2.GaussianBlur(img, (blur_level, blur_level), 0)
    
    # Reduce brightness
    darkened_img = cv2.convertScaleAbs(blurred_img, alpha=brightness_factor, beta=0)
    
    # Save the degraded image if path is provided
    if save_path:
        cv2.imwrite(save_path, darkened_img)
        print(f"Degraded image saved to {save_path}")
    
    return darkened_img

def compare_and_visualize(original_path, degraded_img, show_sift_comparison=True):
    """
    Compare original and degraded images and visualize SIFT keypoints
    
    Parameters:
    original_path (str): Path to the original image
    degraded_img (numpy.ndarray): Degraded image
    show_sift_comparison (bool): Whether to show SIFT keypoint comparison
    """
    # Read the original image
    original_img = cv2.imread(original_path)
    if original_img is None:
        print(f"Error: Could not read original image from {original_path}")
        return
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints in both images
    if show_sift_comparison:
        # Convert to grayscale for SIFT
        gray_original = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        gray_degraded = cv2.cvtColor(degraded_img, cv2.COLOR_BGR2GRAY)
        
        # Detect SIFT keypoints
        kp_original, _ = sift.detectAndCompute(gray_original, None)
        kp_degraded, _ = sift.detectAndCompute(gray_degraded, None)
        
        # Draw keypoints
        img_original_kp = cv2.drawKeypoints(original_img, kp_original, None, 
                                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img_degraded_kp = cv2.drawKeypoints(degraded_img, kp_degraded, None, 
                                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Visualize
        plt.figure(figsize=(16, 8))
        
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(cv2.cvtColor(degraded_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Degraded Image (Blur: {blur_level}, Brightness: {brightness_factor})")
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(img_original_kp, cv2.COLOR_BGR2RGB))
        plt.title(f"Original - SIFT Keypoints: {len(kp_original)}")
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(img_degraded_kp, cv2.COLOR_BGR2RGB))
        plt.title(f"Degraded - SIFT Keypoints: {len(kp_degraded)}")
        plt.axis('off')
        
        print(f"Original image SIFT keypoints: {len(kp_original)}")
        print(f"Degraded image SIFT keypoints: {len(kp_degraded)}")
        print(f"Keypoint reduction: {100 - (len(kp_degraded) / len(kp_original) * 100):.2f}%")
    else:
        # Simple comparison without SIFT keypoints
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(degraded_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Degraded Image (Blur: {blur_level}, Brightness: {brightness_factor})")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('comparison_result.jpg', dpi=300)
    plt.show()

def degrade_images_for_object_detection(object_path, scene_path, blur_level=15, brightness_factor=0.4):
    """
    Degrade both object and scene images for SIFT object detection testing
    
    Parameters:
    object_path (str): Path to the object image
    scene_path (str): Path to the scene image
    blur_level (int): Gaussian blur kernel size
    brightness_factor (float): Factor to reduce brightness
    
    Returns:
    tuple: Paths to degraded object and scene images
    """
    # Generate output filenames
    object_filename = object_path.split('/')[-1]
    scene_filename = scene_path.split('/')[-1]
    
    object_name, object_ext = object_filename.rsplit('.', 1)
    scene_name, scene_ext = scene_filename.rsplit('.', 1)
    
    degraded_object_path = f"{object_name}_degraded.{object_ext}"
    degraded_scene_path = f"{scene_name}_degraded.{scene_ext}"
    
    # Degrade images
    print("Degrading object image...")
    degrade_image(object_path, blur_level, brightness_factor, degraded_object_path)
    
    print("Degrading scene image...")
    degrade_image(scene_path, blur_level, brightness_factor, degraded_scene_path)
    
    print(f"Degraded object image: {degraded_object_path}")
    print(f"Degraded scene image: {degraded_scene_path}")
    
    return degraded_object_path, degraded_scene_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Degrade image quality to test SIFT limitations')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--output', type=str, default=None, help='Path to save the degraded image')
    parser.add_argument('--blur', type=int, default=15, help='Blur level (odd number)')
    parser.add_argument('--brightness', type=float, default=0.4, help='Brightness factor (0-1)')
    parser.add_argument('--no-sift', action='store_true', help='Skip SIFT keypoint comparison')
    
    args = parser.parse_args()
    
    # Global variables for visualization
    blur_level = args.blur
    brightness_factor = args.brightness
    
    # Set default output path if not provided
    if args.output is None:
        image_filename = args.image_path.split('/')[-1]
        name, ext = image_filename.rsplit('.', 1)
        args.output = f"{name}_degraded.{ext}"
    
    # Process the image
    degraded_img = degrade_image(args.image_path, args.blur, args.brightness, args.output)
    
    if degraded_img is not None:
        compare_and_visualize(args.image_path, degraded_img, not args.no_sift)