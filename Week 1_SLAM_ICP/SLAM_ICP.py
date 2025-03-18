import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors

# Definition of the environment (closed space)
environment = np.array([[0, 0], [15, 0], [30, 30], [0, 10], [0, 0]])

# LiDAR simulation (360-degree scan)
def simulate_lidar_scan(robot_position, num_rays=30, max_range=40, noise_std=0.2):
    angles = np.linspace(0, 2*np.pi, num_rays)
    scan_points = []
    
    for angle in angles:
        ray_direction = np.array([np.cos(angle), np.sin(angle)])
        closest_intersection = None
        min_distance = max_range
        
        for i in range(len(environment) - 1):
            p1, p2 = environment[i], environment[i + 1]
            edge_vector = p2 - p1
            t_matrix = np.array([edge_vector, -ray_direction]).T
            if np.linalg.det(t_matrix) == 0:
                continue
            
            t_values = np.linalg.solve(t_matrix, robot_position - p1)
            edge_t, ray_t = t_values
            
            if 0 <= edge_t <= 1 and 0 <= ray_t <= max_range:
                intersection = p1 + edge_t * edge_vector
                if ray_t < min_distance:
                    min_distance = ray_t
                    closest_intersection = intersection
        
        if closest_intersection is not None:
            noise = np.random.normal(0, noise_std, size=2)
            scan_points.append(closest_intersection + noise)
    
    return np.array(scan_points)

# ICP algorithm
def icp(A, B, max_iterations=10, tolerance=1e-4):
    src = np.copy(A)
    prev_error = float('inf')
    
    for i in range(max_iterations):
        nbrs = NearestNeighbors(n_neighbors=1).fit(B)
        distances, indices = nbrs.kneighbors(src)
        
        T = best_fit_transform(src, B[indices.flatten()])
        src = apply_transform(src, T)
        
        mean_error = np.mean(distances)
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
        
    return T

def best_fit_transform(A, B):
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    H = (A - centroid_A).T @ (B - centroid_B)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    t = centroid_B - R @ centroid_A
    return np.hstack((R, t.reshape(-1, 1)))

def apply_transform(points, T):
    return (T[:, :2] @ points.T + T[:, 2].reshape(2, 1)).T

# Estimation of environmental boundaries using α-Shape (improved version)
def alpha_shape(points, alpha=0.5):
    """ Improved α-Shape to better reflect LiDAR point clouds """
    if len(points) < 4:
        return points  # Return as is if there are too few points
    
    tri = Delaunay(points)
    edges = set()
    
    def add_edge(i, j):
        """ Add only unique edges """
        if (j, i) in edges:
            edges.remove((j, i))
        else:
            edges.add((i, j))
    
    for simplex in tri.simplices:
        for i in range(3):
            i1, i2 = simplex[i], simplex[(i+1) % 3]
            add_edge(i1, i2)
    
    edge_points = np.array([points[list(edge)[0]] for edge in edges])

    # **Sort edges by distance to form a continuous boundary**
    center = np.mean(edge_points, axis=0)
    angles = np.arctan2(edge_points[:, 1] - center[1], edge_points[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    sorted_points = edge_points[sorted_indices]

    # **Close the boundary loop**
    sorted_points = np.vstack([sorted_points, sorted_points[0]])

    return sorted_points

# Simulation loop
true_positions, estimated_positions = [], []
aligned_lidar_scans = []
x, y, theta = 1.0, 1.0, 0.0
velocity = 3
angular_velocity = np.pi/30
num_steps = 5

for i in range(num_steps):
    x += velocity * np.cos(theta)
    y += velocity * np.sin(theta)
    theta += angular_velocity
    true_positions.append([x, y])
    
    lidar_scan = simulate_lidar_scan(np.array([x, y]))
    
    if len(estimated_positions) == 0:
        estimated_positions.append([x, y])
        aligned_lidar_scans.append(lidar_scan)
    else:
        prev_position = np.array(estimated_positions[-1])
        prev_scan = aligned_lidar_scans[-1]
        
        transform = icp(lidar_scan, prev_scan)
        estimated_pos = apply_transform(np.array([[x, y]]), transform).flatten()
        estimated_positions.append(estimated_pos.tolist())
        
        aligned_scan = apply_transform(lidar_scan, transform)
        aligned_lidar_scans.append(aligned_scan)

true_positions = np.array(true_positions)
estimated_positions = np.array(estimated_positions)

# **Estimate environmental boundaries**
scan_points_all = np.vstack(aligned_lidar_scans)
optimized_environment = alpha_shape(scan_points_all, alpha=0.3)  # Adjust α value

# **Plot results**
plt.figure(figsize=(10, 8))

# **Plot true environment**
plt.plot(environment[:, 0], environment[:, 1], 'r--', alpha=0.7, linewidth=2, label="True Environment")

# **Plot estimated environment boundary**
plt.plot(optimized_environment[:, 0], optimized_environment[:, 1], 'k--', linewidth=2, label="Estimated Environment")

# **Plot LiDAR point clouds**
for scan in aligned_lidar_scans:
    plt.scatter(scan[:, 0], scan[:, 1], s=20, c='gray', alpha=0.5, label="LiDAR Points" if scan is aligned_lidar_scans[0] else "")

# **Plot robot trajectory**
plt.plot(true_positions[:, 0], true_positions[:, 1], 'bo-', label="Ground Truth Trajectory")
plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], 'ro-', marker='x', linestyle='-', label="Estimated Trajectory")

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")
plt.title("2D LiDAR-based SLAM")
plt.grid()
plt.show()