import cv2
import numpy as np
from sklearn.decomposition import PCA

def generate_ellipse_image():
    height, weight = 400, 400
    image = np.ones((height, weight, 3), np.uint8) * 255
    
    center = (weight // 3, height // 3)
    axes = (50, 30)
    angle = 20
    start_angle = 0
    end_angle = 360
    color = (255, 0, 0)
    thickness = -1
    
    cv2.ellipse(image, center, axes, angle, start_angle, end_angle, color, thickness)
    
    cv2.imwrite("blue_ellipse.png", image)
    
    return image
    
def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    cv2.imwrite("gray.png", gray)
    
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite("binary.png", binary)
    
    points = np.column_stack(np.where(binary > 0))
    
    # Coordinate transformation: (row, column) -> (column, row)
    points = points[:, [1, 0]]
    print("Extracted points (x, y):", points)
    
    pca = PCA(n_components=2)
    pca.fit(points)
    center = np.mean(points, axis=0)
    principal_axes = pca.components_
    eigenvalues = pca.explained_variance_
    print("center:", center)
    print("principal_axes:", principal_axes)
    print("eigenvalues:", eigenvalues)
    
    colors = [(0, 0, 255), (0, 255, 0)]
    for i, axis in enumerate(principal_axes):
        length = int(eigenvalues[i] * 0.1)
        end_point = center + axis * length
        end_point = tuple(map(int, end_point))
        cv2.line(image, tuple(map(int, center)), end_point, colors[i], 2)
        
    # # Draw contours (yellow outline) (use when there are multiple contours)
    # contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(image, contours, -1, (0, 255, 255), 2)
        
    fixed_length = 20
    offset = 40
    first_axis = principal_axes[0] * fixed_length
    parallel_offset = principal_axes[1] * offset
    
    for sign in [-1, 1]:
        start_point = center + first_axis + sign * parallel_offset
        end_point = center - first_axis + sign * parallel_offset
        start_point = tuple(map(int, start_point))
        end_point = tuple(map(int, end_point))
        cv2.line(image, start_point, end_point, (0, 165, 255), 2)
        
    return image

def main():
    image = generate_ellipse_image()
    processed_image = process_image(image)
    
    # Save and display the result
    cv2.imwrite("processed_ellipse.png", processed_image)
    cv2.imshow("Processed Image", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()