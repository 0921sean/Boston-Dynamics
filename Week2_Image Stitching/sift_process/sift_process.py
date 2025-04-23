import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_gaussian_and_dog(image_path):
    """
    Visualize Gaussian blur at different sigma values and their Difference of Gaussians (DoG)
    
    Parameters:
    image_path (str): Path to the input image
    """
    # Load image and convert to grayscale
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Define sigma values for Gaussian blur
    sigma_values = [1.0, 1.6, 2.0, 3.0, 4.0]
    
    # Create list to store blurred images
    gaussian_images = []
    
    # Apply Gaussian blur with different sigma values
    for sigma in sigma_values:
        # Calculate kernel size based on sigma
        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:  # Ensure kernel size is odd
            ksize += 1
            
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
        gaussian_images.append(blurred)
    
    # Calculate Difference of Gaussians (DoG)
    dog_images = []
    for i in range(len(gaussian_images) - 1):
        dog = cv2.subtract(gaussian_images[i], gaussian_images[i + 1])
        # Normalize DoG image for better visualization
        dog_normalized = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
        dog_images.append(dog_normalized)
    
    # Visualize results
    fig, axes = plt.subplots(2, len(sigma_values), figsize=(15, 6))
    
    # Show original image
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Show Gaussian blurred images
    for i, sigma in enumerate(sigma_values):
        if i > 0:  # Skip the original slot
            axes[0, i].imshow(gaussian_images[i-1], cmap='gray')
            axes[0, i].set_title(f'σ = {sigma_values[i-1]}')
            axes[0, i].axis('off')
    
    # Show DoG images
    for i in range(len(dog_images)):
        axes[1, i+1].imshow(dog_images[i], cmap='gray')
        axes[1, i+1].set_title(f'DoG: σ{i+1} - σ{i+2}')
        axes[1, i+1].axis('off')
    
    # Empty plot for alignment
    axes[1, 0].axis('off')
    
    plt.tight_layout()
    plt.savefig('gaussian_and_dog.png')
    plt.show()
    
    return gaussian_images, dog_images

def find_dog_extrema(dog_images):
    """
    Find extrema in DoG images (simplified version)
    
    Parameters:
    dog_images (list): List of DoG images
    
    Returns:
    numpy.ndarray: Image with extrema points marked
    """
    if len(dog_images) < 3:
        print("Need at least 3 DoG images to find extrema")
        return None
    
    # Create a copy of the middle DoG image to mark extrema points
    result = cv2.cvtColor(dog_images[1].astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    # For simplicity, we'll check a subset of the image
    height, width = dog_images[1].shape
    step = 5  # Check every 5 pixels to speed up processing
    
    for y in range(step, height - step, step):
        for x in range(step, width - step, step):
            # Get the 3x3 neighborhood in the current DoG image
            current_patch = dog_images[1][y-1:y+2, x-1:x+2]
            
            # Get the corresponding patches in adjacent DoG images
            prev_patch = dog_images[0][y-1:y+2, x-1:x+2]
            next_patch = dog_images[2][y-1:y+2, x-1:x+2]
            
            # Current pixel value
            current_val = dog_images[1][y, x]
            
            # Check if it's a maximum across all 3 scales
            is_max = True
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    # Skip comparing to itself in the current scale
                    if dy == 0 and dx == 0:
                        continue
                    
                    # Check current scale
                    if y+dy >= 0 and y+dy < height and x+dx >= 0 and x+dx < width:
                        if current_val <= dog_images[1][y+dy, x+dx]:
                            is_max = False
                            break
                    
                    # Check previous scale
                    if y+dy >= 0 and y+dy < height and x+dx >= 0 and x+dx < width:
                        if current_val <= dog_images[0][y+dy, x+dx]:
                            is_max = False
                            break
                    
                    # Check next scale
                    if y+dy >= 0 and y+dy < height and x+dx >= 0 and x+dx < width:
                        if current_val <= dog_images[2][y+dy, x+dx]:
                            is_max = False
                            break
                
                if not is_max:
                    break
            
            # Check if it's a minimum across all 3 scales
            is_min = True
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    # Skip comparing to itself in the current scale
                    if dy == 0 and dx == 0:
                        continue
                    
                    # Check current scale
                    if y+dy >= 0 and y+dy < height and x+dx >= 0 and x+dx < width:
                        if current_val >= dog_images[1][y+dy, x+dx]:
                            is_min = False
                            break
                    
                    # Check previous scale
                    if y+dy >= 0 and y+dy < height and x+dx >= 0 and x+dx < width:
                        if current_val >= dog_images[0][y+dy, x+dx]:
                            is_min = False
                            break
                    
                    # Check next scale
                    if y+dy >= 0 and y+dy < height and x+dx >= 0 and x+dx < width:
                        if current_val >= dog_images[2][y+dy, x+dx]:
                            is_min = False
                            break
                
                if not is_min:
                    break
            
            # Mark extrema points
            if is_max or is_min:
                cv2.circle(result, (x, y), 2, (0, 0, 255), -1)
    
    return result

# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = "my_image1.jpeg"
    
    # Visualize Gaussian blur and DoG
    gaussian_images, dog_images = visualize_gaussian_and_dog(image_path)
    
    # If at least 3 DoG images are available, find and visualize extrema
    if len(dog_images) >= 3:
        extrema_image = find_dog_extrema(dog_images)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(extrema_image, cv2.COLOR_BGR2RGB))
        plt.title('Potential SIFT Keypoints (DoG Extrema)')
        plt.axis('off')
        plt.savefig('dog_extrema.png')
        plt.show()