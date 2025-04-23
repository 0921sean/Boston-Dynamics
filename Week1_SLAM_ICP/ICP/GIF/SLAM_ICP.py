import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.spatial import KDTree
import datetime

# ----- Function to generate the point cloud of a cuboid surface (density specified version) -----
def create_cuboid_surface(dims, density, origin=(0,0,0)):
    """
    dims: (dx, dy, dz) The lengths of each edge of the cuboid
    density: The point density per unit length (points per unit length)
    origin: The coordinate of one vertex of the cuboid (use (0,0,0) if aligning with the origin)
    
    Generates a point cloud by sampling each of the six faces of the cuboid in a grid pattern,
    and returns the unique points (without duplicates).
    """
    dx, dy, dz = dims
    ox, oy, oz = origin
    points = []

    # Faces 1 and 2: faces on the xy plane (z = oz, z = oz+dz)
    nx = max(2, int(np.ceil(dx * density)))
    ny = max(2, int(np.ceil(dy * density)))
    u = np.linspace(0, dx, nx)
    v = np.linspace(0, dy, ny)
    U, V = np.meshgrid(u, v)
    X = U.flatten() + ox
    Y = V.flatten() + oy
    # Face at z = oz
    Z = np.full_like(X, oz)
    points.append(np.column_stack([X, Y, Z]))
    # Face at z = oz+dz
    Z = np.full_like(X, oz+dz)
    points.append(np.column_stack([X, Y, Z]))

    # Faces 3 and 4: faces on the xz plane (y = oy, y = oy+dy)
    nx = max(2, int(np.ceil(dx * density)))
    nz = max(2, int(np.ceil(dz * density)))
    u = np.linspace(0, dx, nx)
    v = np.linspace(0, dz, nz)
    U, V = np.meshgrid(u, v)
    X = U.flatten() + ox
    Z = V.flatten() + oz
    # Face at y = oy
    Y = np.full_like(X, oy)
    points.append(np.column_stack([X, Y, Z]))
    # Face at y = oy+dy
    Y = np.full_like(X, oy+dy)
    points.append(np.column_stack([X, Y, Z]))

    # Faces 5 and 6: faces on the yz plane (x = ox, x = ox+dx)
    ny = max(2, int(np.ceil(dy * density)))
    nz = max(2, int(np.ceil(dz * density)))
    u = np.linspace(0, dy, ny)
    v = np.linspace(0, dz, nz)
    U, V = np.meshgrid(u, v)
    Y = U.flatten() + oy
    Z = V.flatten() + oz
    # Face at x = ox
    X = np.full_like(Y, ox)
    points.append(np.column_stack([X, Y, Z]))
    # Face at x = ox+dx
    X = np.full_like(Y, ox+dx)
    points.append(np.column_stack([X, Y, Z]))

    points = np.vstack(points)
    # Remove duplicate points
    points = np.unique(points, axis=0)
    return points

# ----- Generate a random rotation matrix -----
def random_rotation_matrix():
    angle = np.random.uniform(0, 2*np.pi)
    axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    x, y, z = axis
    R = np.array([
        [c + x*x*C,    x*y*C - z*s,  x*z*C + y*s],
        [y*x*C + z*s,  c + y*y*C,    y*z*C - x*s],
        [z*x*C - y*s,  z*y*C + x*s,  c + z*z*C]
    ])
    return R

# ----- Implementation of the ICP algorithm -----
def icp(A, B, max_iterations=100, tolerance=1e-6):
    """
    A: Point cloud of the pre-scanned model (Nx3) → subject to be transformed
    B: Point cloud of the segmented model (Nx3) → fixed
    max_iterations: Maximum number of iterations
    tolerance: Convergence threshold (stop if the mean nearest neighbor distance is below this value)
    
    Returns:
      - A_trans: A after the final transformation
      - frames: The state of A at each iteration (for visualization)
    """
    A_trans = A.copy()
    frames = [A_trans.copy()]
    tree = KDTree(B)
    
    for itr in range(max_iterations):
        distances, indices = tree.query(A_trans)
        B_corr = B[indices]

        centroid_A = np.mean(A_trans, axis=0)
        centroid_B = np.mean(B_corr, axis=0)
        
        AA = A_trans - centroid_A
        BB = B_corr - centroid_B
        
        H = AA.T @ BB
        U, S, Vt = np.linalg.svd(H)
        R_icp = Vt.T @ U.T
        if np.linalg.det(R_icp) < 0:
            Vt[2, :] *= -1
            R_icp = Vt.T @ U.T
        t_icp = centroid_B - R_icp @ centroid_A
        
        A_trans = (R_icp @ A_trans.T).T + t_icp
        frames.append(A_trans.copy())
        
        mean_error = np.mean(distances)
        print(f"Iteration {itr+1}: mean error = {mean_error:.6f}")
        if mean_error < tolerance:
            break
    return A_trans, frames

# ----- Main process -----
if __name__ == '__main__':
    # Set parameters
    dims = (2, 5, 8)
    density = 10  # Point density per unit length
    max_iter = 50
    
    # Pre-scanned model: generate a point cloud on the surface of a cuboid
    # aligned with the x, y, z axes, with one vertex at the origin
    pre_model = create_cuboid_surface(dims, density=density, origin=(0,0,0))
    
    # Segmented model: same size, but applies a random rotation and translation
    R_rand = random_rotation_matrix()
    t_rand = np.random.uniform(-5, 5, 3)
    segmented_model = (R_rand @ pre_model.T).T + t_rand
    
    # Run ICP: align pre_model to segmented_model
    final_model, frames = icp(pre_model, segmented_model, max_iterations=max_iter, tolerance=1e-6)
    
    # Set visualization range
    all_points = np.vstack((segmented_model, pre_model))
    x_min, y_min, z_min = np.min(all_points, axis=0) - 1
    x_max, y_max, z_max = np.max(all_points, axis=0) + 1
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame_idx):
        ax.cla()
        # Segmented model remains fixed (red)
        ax.scatter(segmented_model[:, 0], segmented_model[:, 1], segmented_model[:, 2],
                   c='r', s=0.05, label='Segmented Model')
        # Pre-scanned model transformed by ICP (blue)
        current_points = frames[frame_idx]
        ax.scatter(current_points[:, 0], current_points[:, 1], current_points[:, 2],
                   c='b', s=0.05, label='Pre-scanned Model (Transformed)')
        ax.set_title(f"ICP Iteration: {frame_idx}")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
    
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=300, repeat=False)
    
    # Get current date and time, convert to a string in YYYYMMDD_HHMM format
    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M")
    
    # Save video: include date and time in the filename
    filename_mp4 = f'icp_alignment_{date_str}.mp4'
    filename_gif = f'icp_alignment_{date_str}.gif'
    
    # try:
    #     ani.save(filename_mp4, writer='ffmpeg', fps=5)
    #     print(f"Video saved as {filename_mp4}")
    # except Exception as e:
    #     print("Failed to save with ffmpeg, saving as GIF instead.", e)
    #     ani.save(filename_gif, writer='pillow', fps=5)
    #     print(f"Video saved as {filename_gif}")
        
    ani.save('./GIF/'+filename_gif, writer='pillow', fps=5)
    print(f"Video saved as {filename_gif}")
    
    plt.show()
