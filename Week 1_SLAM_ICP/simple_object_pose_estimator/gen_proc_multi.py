import cv2
import numpy as np
from sklearn.decomposition import PCA

def generate_ellipse_image():
    height, width = 400, 400
    image = np.ones((height, width, 3), np.uint8) * 255
    
    # List of ellipse parameters
    ellipses = [
        ((width // 3, height // 3), (50, 30), 20),  # First ellipse: (133, 133), major axis 100, minor axis 60, angle 20 degrees
        ((2 * width // 3, height // 3), (40, 20), 45),  # Second ellipse: (266, 133), major axis 80, minor axis 40, angle 45 degrees
        ((width // 3, 2 * height // 3), (30, 15), 60)  # Third ellipse: (133, 266), major axis 60, minor axis 30, angle 60 degrees
    ]
    
    for center, axes, angle in ellipses:
        color = (255, 0, 0)  # Blue
        thickness = -1
        cv2.ellipse(image, center, axes, angle, 0, 360, color, thickness)
    
    cv2.imwrite("blue_ellipses.png", image)
    # Remove window display
    return image, ellipses

def process_image(image, ellipses):
    for center, axes, angle in ellipses:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Extract points for each ellipse from binary image (using approximate ROI)
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
        binary_masked = cv2.bitwise_and(binary, binary, mask=mask)
        points = np.column_stack(np.where(binary_masked > 0))
        points = points[:, [1, 0]]  # Convert to (column, row) format
        print(f"Extracted points (x, y) for center {center}:", points)
        
        pca = PCA(n_components=2)
        pca.fit(points)
        center_point = np.mean(points, axis=0)  # (x, y) format
        principal_axes = pca.components_
        eigenvalues = pca.explained_variance_
        print(f"center: {center_point}")
        print(f"principal_axes: {principal_axes}")
        print(f"eigenvalues: {eigenvalues}")
        
        # Visualize principal axes
        colors = [(0, 0, 255), (0, 255, 0)]  # Red (major axis), Green (minor axis)
        for i, axis in enumerate(principal_axes):
            length = int(eigenvalues[i] * 0.1)
            # length = axes[i]  # Major axis: axes[0], Minor axis: axes[1]
            end_point = center_point + axis * length
            end_point = tuple(map(int, end_point))
            cv2.line(image, tuple(map(int, center_point)), end_point, colors[i], 2)
        
        # Draw contours (yellow outline)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(image, contours, -1, (0, 255, 255), 2)
        
        # Parallel jaw gripper
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
    image, ellipses = generate_ellipse_image()
    processed_image = process_image(image, ellipses)
    
    # Save and display the result
    cv2.imwrite("processed_ellipses.png", processed_image)
    cv2.imshow("Processed Image", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()