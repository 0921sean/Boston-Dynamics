import cv2
import numpy as np
from sklearn.decomposition import PCA

def process_custom_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Create a copy of the image (preserve original)
    processed_image = image.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Image binarization (threshold may need adjustment)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process each contour
    for contour in contours:
        # Ignore small noise
        if cv2.contourArea(contour) < 100:
            continue
        
        # Extract points from contour
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        points = np.column_stack(np.where(mask > 0))
        points = points[:, [1, 0]]  # Convert to (column, row) format
        
        # Check if there are enough points
        if len(points) < 5:
            continue
            
        print(f"Contour points: {len(points)}")
        
        # Apply PCA
        pca = PCA(n_components=2)
        pca.fit(points)
        center_point = np.mean(points, axis=0)  # (x, y) format
        principal_axes = pca.components_
        eigenvalues = pca.explained_variance_
        
        print(f"Center: {center_point}")
        print(f"Principal axes: {principal_axes}")
        print(f"Eigenvalues: {eigenvalues}")
        
        # Draw contour (yellow)
        cv2.drawContours(processed_image, [contour], -1, (0, 255, 255), 2)
        
        # Visualize principal axes
        colors = [(0, 0, 255), (0, 255, 0)]  # Red (major axis), Green (minor axis)
        for i, axis in enumerate(principal_axes):
            length = int(np.sqrt(eigenvalues[i]) * 3)  # Scale adjustment
            end_point = center_point + axis * length
            start_point = tuple(map(int, center_point))
            end_point = tuple(map(int, end_point))
            cv2.line(processed_image, start_point, end_point, colors[i], 2)
        
        # Visualize parallel jaw gripper
        fixed_length = int(np.sqrt(eigenvalues[0]) * 2)  # Gripper length based on major axis
        offset = int(np.sqrt(eigenvalues[1]) * 2)  # Gripper offset based on minor axis
        
        first_axis = principal_axes[0] * fixed_length
        parallel_offset = principal_axes[1] * offset
        
        for sign in [-1, 1]:
            start_point = center_point + first_axis + sign * parallel_offset
            end_point = center_point - first_axis + sign * parallel_offset
            start_point = tuple(map(int, start_point))
            end_point = tuple(map(int, end_point))
            cv2.line(processed_image, start_point, end_point, (0, 165, 255), 2)
    
    return processed_image

def generate_ellipse_image():
    # Keep the original ellipse generation code
    height, width = 400, 400
    image = np.ones((height, width, 3), np.uint8) * 255
    
    ellipses = [
        ((width // 3, height // 3), (50, 30), 20),
        ((2 * width // 3, height // 3), (40, 20), 45),
        ((width // 3, 2 * height // 3), (30, 15), 60)
    ]
    
    for center, axes, angle in ellipses:
        color = (255, 0, 0)  # Blue
        thickness = -1
        cv2.ellipse(image, center, axes, angle, 0, 360, color, thickness)
    
    cv2.imwrite("blue_ellipses.png", image)
    return image, ellipses

def process_ellipse_image(image, ellipses):
    # Keep the original ellipse processing code
    for center, axes, angle in ellipses:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
        binary_masked = cv2.bitwise_and(binary, binary, mask=mask)
        points = np.column_stack(np.where(binary_masked > 0))
        points = points[:, [1, 0]]
        
        pca = PCA(n_components=2)
        pca.fit(points)
        center_point = np.mean(points, axis=0)
        principal_axes = pca.components_
        eigenvalues = pca.explained_variance_
        
        colors = [(0, 0, 255), (0, 255, 0)]
        for i, axis in enumerate(principal_axes):
            length = int(eigenvalues[i] * 0.1)
            end_point = center_point + axis * length
            end_point = tuple(map(int, end_point))
            cv2.line(image, tuple(map(int, center_point)), end_point, colors[i], 2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(image, contours, -1, (0, 255, 255), 2)
        
        fixed_length = 20
        offset = 40
        first_axis = principal_axes[0] * fixed_length
        parallel_offset = principal_axes[1] * offset
        
        for sign in [-1, 1]:
            start_point = center_point + first_axis + sign * parallel_offset
            end_point = center_point - first_axis + sign * parallel_offset
            start_point = tuple(map(int, start_point))
            end_point = tuple(map(int, end_point))
            cv2.line(image, start_point, end_point, (0, 165, 255), 2)
    
    return image

def main():
    # 1. Process custom image (use ellipse image as default)
    use_custom_image = True
    custom_image_path = "colorful_objects.jpg"  # Enter your desired image path here
    
    if use_custom_image:
        try:
            processed_image = process_custom_image(custom_image_path)
            result_path = "processed_custom_image.png"
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Generating ellipse image instead.")
            use_custom_image = False
    
    # 2. Process ellipse image (if custom image is not available or error occurs)
    if not use_custom_image:
        image, ellipses = generate_ellipse_image()
        processed_image = process_ellipse_image(image, ellipses)
        result_path = "processed_ellipses.png"
    
    # Save and display
    cv2.imwrite(result_path, processed_image)
    cv2.imshow("Processed Image", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()