import os
import numpy as np
import scipy.ndimage
import trimesh
from scipy.spatial.transform import Rotation as R

# Define base paths
ycb_grasp_dataset_dir = "/home/haoming/Downloads/ycb_grasp_dataset"
input_dir = os.path.join(ycb_grasp_dataset_dir, "input")
gt_dir = os.path.join(ycb_grasp_dataset_dir, "gt")
meshes_dir = "/home/haoming/Downloads/ycb_meshes/ycb-objects/meshes"

# Function to generate a random rotation matrix
def random_rotation_matrix():
    # Generate random rotation matrix using random Euler angles
    return R.random().as_matrix()

# Function to apply rotation to the mesh vertices
def rotate_mesh(mesh, rotation_matrix):
    # Apply rotation to mesh vertices by multiplying with the rotation matrix
    mesh.vertices = mesh.vertices @ rotation_matrix.T
    return mesh

# Function to normalize the mesh vertices to range [-0.5, 0.5]
def normalize_mesh(mesh):
    # Get the bounding box of the vertices
    min_bounds = mesh.bounds[0]
    max_bounds = mesh.bounds[1]

    # Compute the center of the bounding box
    center = (min_bounds + max_bounds) / 2.0

    # Shift the vertices to be centered at the origin
    mesh.vertices -= center

    # Compute the scale factor to fit the vertices in [-0.5, 0.5]
    scale = np.max(max_bounds - min_bounds)

    # Scale the vertices to fit within the range [-0.5, 0.5]
    mesh.vertices /= scale

    return mesh

def force_cubic_normalization(mesh):
    min_bounds = mesh.bounds[0]
    max_bounds = mesh.bounds[1]

    # Compute the center of the bounding box
    center = (min_bounds + max_bounds) / 2.0

    # Compute the ranges along each axis
    ranges = max_bounds - min_bounds

    # Find the maximum range across all axes
    max_range = np.max(ranges)

    # Scale each axis proportionally to make the mesh cubic
    scales = max_range / ranges

    # Translate the mesh to the center
    mesh.apply_translation(-center)

    # Apply non-uniform scaling to make the bounding box cubic
    mesh.apply_scale(scales)

    # Finally, apply uniform scaling to fit the bounding box in [-0.5, 0.5]
    mesh.apply_scale(1 / max_range)

    return mesh

# Function to sample a partial point cloud from the complete point cloud
def sample_partial_from_mesh(mesh, num_points=2048):
    """
    This function takes the complete point cloud and samples 2048 points
    from the bottom half (where z < 0).
    """
    # Extract the bottom half points where the z-coordinate is less than 0
    sampled_full_points, _ = trimesh.sample.sample_surface(mesh, num_points * 8)
    bottom_half_points = sampled_full_points[sampled_full_points[:, 2] < 0]

    # If there are enough points, randomly sample 2048, otherwise take all
    if len(bottom_half_points) > num_points:
        sampled_points = bottom_half_points[np.random.choice(len(bottom_half_points), num_points, replace=False)]
    else:
        repeat_indices = np.random.choice(len(bottom_half_points), num_points - len(bottom_half_points), replace=True)
        sampled_points = np.vstack([bottom_half_points, bottom_half_points[repeat_indices]])

    return sampled_points

# Function to sample a complete point cloud from the mesh
def sample_complete_from_mesh(mesh, num_points=8192):
    # Sample uniformly from the mesh surface
    sampled_points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return sampled_points

# Function to create a solid occupancy grid from the mesh
def create_solid_occupancy_grid(mesh, resolution=32):
    # Create a voxelized version of the mesh
    voxel_grid = mesh.voxelized(pitch=1.0 / resolution).matrix

    # Fill hollow parts inside the mesh to make the occupancy grid solid
    voxel_grid = scipy.ndimage.binary_fill_holes(voxel_grid)
    #voxel_grid = scipy.ndimage.binary_dilation(voxel_grid, iterations=2)

    return voxel_grid.astype(int)

# Function to save a point cloud to an XYZ file
def save_xyz(points, file_path):
    np.savetxt(file_path, points, fmt="%.6f")

# Function to process each mesh and generate rotated variants
def process_mesh_variants(mesh_path, category_name):
    # Load and normalize the mesh
    mesh = trimesh.load(mesh_path)
    #normalized_mesh = normalize_mesh(mesh)
    #normalized_mesh = force_cubic_normalization(mesh)

    # Create directories for train and test
    train_input_dir = os.path.join(input_dir, category_name, "train")
    test_input_dir = os.path.join(input_dir, category_name, "test")
    train_gt_dir = os.path.join(gt_dir, category_name, "train")
    test_gt_dir = os.path.join(gt_dir, category_name, "test")

    os.makedirs(train_input_dir, exist_ok=True)
    os.makedirs(test_input_dir, exist_ok=True)
    os.makedirs(train_gt_dir, exist_ok=True)
    os.makedirs(test_gt_dir, exist_ok=True)

    # Generate 700 variants using random rotations
    for i in range(700):
        suffix = f"_{i}"

        # Generate a random rotation matrix
        rotation_matrix = random_rotation_matrix()

        # Rotate the mesh before sampling point clouds and generating the occupancy grid
        rotated_mesh = rotate_mesh(mesh.copy(), rotation_matrix)
        normalized_mesh = force_cubic_normalization(rotated_mesh)

        # Sample the complete and partial point clouds from the rotated mesh
        rotated_complete = sample_complete_from_mesh(normalized_mesh)
        rotated_partial = sample_partial_from_mesh(normalized_mesh)
        rotated_occupancy_grid = create_solid_occupancy_grid(normalized_mesh)

        # Save partial and complete point clouds
        partial_variant_path = os.path.join(train_input_dir if i < 600 else test_input_dir, f"{category_name}{suffix}_x.xyz")
        complete_variant_path = os.path.join(train_gt_dir if i < 600 else test_gt_dir, f"{category_name}{suffix}_y.xyz")
        save_xyz(rotated_partial, partial_variant_path)
        save_xyz(rotated_complete, complete_variant_path)

        # Save the rotated occupancy grid
        occupancy_variant_path = os.path.join(train_gt_dir if i < 600 else test_gt_dir, f"{category_name}{suffix}.npy")
        np.save(occupancy_variant_path, rotated_occupancy_grid)

        print(f"Processed variant {i + 1}/700 for category {category_name}")

# Main function to process all meshes
def main():
    mesh_files = [f for f in os.listdir(meshes_dir) if f.endswith(".ply")]
    counter = 0

    for mesh_file in mesh_files:
        category_name = mesh_file.replace(".ply", "")
        mesh_path = os.path.join(meshes_dir, mesh_file)

        # Process the set of files for this category
        process_mesh_variants(mesh_path, category_name)
        counter = counter + 1
        print(f"========================================= {counter}/ 83 ==============================================")

if __name__ == "__main__":
    main()


