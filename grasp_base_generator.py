import os
import numpy as np
import trimesh
import scipy.ndimage
from scipy.spatial.transform import Rotation as R

# Define the directories
base_dir = "/home/haoming/Downloads/ycb_meshes/grasp_database"
output_dir = "/home/haoming/Downloads/ycb_meshes/grasp_database/processed"
complete_output_dir = os.path.join(output_dir, "complete_pcs")
partial_output_dir = os.path.join(output_dir, "partial_pcs")
occupancy_output_dir = os.path.join(output_dir, "occupancy_grids")
variant_output_dir = os.path.join(output_dir, "variants")

# Create output directories if they don't exist
os.makedirs(complete_output_dir, exist_ok=True)
os.makedirs(partial_output_dir, exist_ok=True)
os.makedirs(occupancy_output_dir, exist_ok=True)
os.makedirs(variant_output_dir, exist_ok=True)

# Number of points for complete and partial point clouds
num_complete_points = 8192
num_partial_points = 2048
num_variants = 700
voxel_resolution = 32

# Function to normalize the mesh vertices to range [-0.5, 0.5]
def normalize_mesh_vertices(mesh):
    min_bounds = mesh.bounds[0]
    max_bounds = mesh.bounds[1]
    center = (min_bounds + max_bounds) / 2.0
    mesh.vertices -= center
    scale = np.max(max_bounds - min_bounds)
    mesh.vertices /= scale

# Function to load a PLY file as a trimesh object
def load_mesh(mesh_path):
    return trimesh.load(mesh_path)

# Function to generate complete and partial point clouds
def generate_point_clouds(mesh):
    sampled_points, _ = trimesh.sample.sample_surface(mesh, num_complete_points)
    partial_points = sampled_points[sampled_points[:, 2] < 0]
    if len(partial_points) < num_partial_points:
        return sampled_points, partial_points
    else:
        partial_indices = np.random.choice(len(partial_points), num_partial_points, replace=False)
        return sampled_points, partial_points[partial_indices]

# Function to create the occupancy grid
def generate_occupancy_grid(mesh):
    voxelized_mesh = mesh.voxelized(pitch=1.0 / voxel_resolution)
    return voxelized_mesh.matrix.astype(int)

# Function to save point clouds and occupancy grids
def save_xyz(file_path, points):
    np.savetxt(file_path, points, fmt='%.6f')

def save_occupancy_grid(file_path, occupancy_grid):
    np.save(file_path, occupancy_grid)

# Function to generate a random 3D rotation matrix
def random_rotation_matrix():
    return R.random().as_matrix()

# Function to apply rotation to the point clouds and occupancy grid
def apply_rotation(points, rotation_matrix):
    return points @ rotation_matrix.T

def apply_rotation_to_occupancy_grid(grid, rotation_matrix):
    rotated_grid = scipy.ndimage.rotate(grid, angle=np.rad2deg(np.arccos(rotation_matrix[0, 0])), axes=(0, 1), reshape=False)
    return rotated_grid

# Function to process each mesh and generate its variants
def process_mesh(mesh_name, mesh_path):
    # Load and normalize the mesh
    mesh = load_mesh(mesh_path)
    normalize_mesh_vertices(mesh)

    # Generate complete and partial point clouds
    complete_points, partial_points = generate_point_clouds(mesh)

    # Generate occupancy grid
    occupancy_grid = generate_occupancy_grid(mesh)

    # Save the original point clouds and occupancy grid
    complete_file = os.path.join(complete_output_dir, f"{mesh_name}_complete.xyz")
    partial_file = os.path.join(partial_output_dir, f"{mesh_name}_partial.xyz")
    occupancy_file = os.path.join(occupancy_output_dir, f"{mesh_name}_occupancy.npy")

    save_xyz(complete_file, complete_points)
    save_xyz(partial_file, partial_points)
    save_occupancy_grid(occupancy_file, occupancy_grid)

    # Generate variants
    for i in range(num_variants):
        rotation_matrix = random_rotation_matrix()

        rotated_complete_points = apply_rotation(complete_points, rotation_matrix)
        rotated_partial_points = apply_rotation(partial_points, rotation_matrix)
        rotated_occupancy_grid = apply_rotation_to_occupancy_grid(occupancy_grid, rotation_matrix)

        variant_complete_file = os.path.join(variant_output_dir, f"{mesh_name}_complete_variant_{i}.xyz")
        variant_partial_file = os.path.join(variant_output_dir, f"{mesh_name}_partial_variant_{i}.xyz")
        variant_occupancy_file = os.path.join(variant_output_dir, f"{mesh_name}_occupancy_variant_{i}.npy")

        save_xyz(variant_complete_file, rotated_complete_points)
        save_xyz(variant_partial_file, rotated_partial_points)
        save_occupancy_grid(variant_occupancy_file, rotated_occupancy_grid)

        print(f"Created variant {i+1} for {mesh_name}")

# Function to process all meshes in the dataset
def process_dataset():
    # Loop through each folder in the grasp_database
    for folder in os.listdir(base_dir):
        mesh_folder = os.path.join(base_dir, folder, "meshes")
        if os.path.isdir(mesh_folder):
            mesh_file = os.path.join(mesh_folder, f"{folder}_scaled.ply")
            if os.path.isfile(mesh_file):
                print(f"Processing mesh {folder}")
                process_mesh(folder, mesh_file)

# Run the processing
if __name__ == "__main__":
    process_dataset()
