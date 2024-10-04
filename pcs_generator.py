import os
import trimesh
import numpy as np

# Define the directories for input and output
mesh_directory = "/home/haoming/Downloads/ycb_meshes/ycb-objects/meshes"
complete_output_directory = "/home/haoming/Downloads/ycb_meshes/ycb-objects/complete_pcs"
partial_output_directory = "/home/haoming/Downloads/ycb_meshes/ycb-objects/partial_pcs"
occupancy_output_directory = "/home/haoming/Downloads/ycb_meshes/ycb-objects/occupancy_grids"

# Create output directories if they don't exist
os.makedirs(complete_output_directory, exist_ok=True)
os.makedirs(partial_output_directory, exist_ok=True)
os.makedirs(occupancy_output_directory, exist_ok=True)

# Number of points for complete and partial point clouds
num_complete_points = 8192
num_partial_points = 2048

# Voxel resolution for the occupancy grid
voxel_resolution = 32


# Function to normalize the mesh vertices to range [-0.5, 0.5]
def normalize_mesh_vertices(mesh):
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


# Function to sample points from the bottom half of the point cloud
def sample_partial_from_bottom(points, num_partial_points):
    # Select points from the bottom half (based on the z-axis)
    bottom_half_points = points[points[:, 2] < 0]

    # If there are not enough points in the bottom half, adjust the number
    if len(bottom_half_points) < num_partial_points:
        return bottom_half_points
    else:
        # Randomly sample 2048 points from the bottom half
        return bottom_half_points[np.random.choice(bottom_half_points.shape[0], num_partial_points, replace=False)]


# Function to create and save an occupancy grid
def create_and_save_occupancy_grid(mesh, output_path, voxel_resolution):
    # Create the voxelized version of the mesh
    voxelized_mesh = mesh.voxelized(pitch=1.0 / voxel_resolution)

    # Get the occupancy grid (as a numpy array)
    occupancy_grid = voxelized_mesh.matrix.astype(int)

    # Save the occupancy grid as a .npy file
    np.save(output_path, occupancy_grid)


# Function to sample points from the normalized mesh surface, and save as xyz files
def sample_and_save_points_as_xyz(mesh, complete_output_path, partial_output_path, num_complete_points,
                                  num_partial_points):
    # Sample points uniformly on the surface of the normalized mesh
    sampled_points, _ = trimesh.sample.sample_surface(mesh, num_complete_points)

    # Save the complete point cloud to the .xyz file
    with open(complete_output_path, 'w') as f:
        for point in sampled_points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

    # Generate the partial point cloud from the bottom half and save
    partial_points = sample_partial_from_bottom(sampled_points, num_partial_points)
    with open(partial_output_path, 'w') as f:
        for point in partial_points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")


# Iterate through all files in the directory
for filename in os.listdir(mesh_directory):
    if filename.endswith(".ply"):
        # Full path to the .ply file
        ply_file_path = os.path.join(mesh_directory, filename)

        # Output file names for complete, partial point clouds, and occupancy grid
        complete_output_file_path = os.path.join(complete_output_directory, filename.replace(".ply", ".xyz"))
        partial_output_file_path = os.path.join(partial_output_directory, filename.replace(".ply", ".xyz"))
        occupancy_output_file_path = os.path.join(occupancy_output_directory, filename.replace(".ply", ".npy"))

        # Load the mesh
        mesh = trimesh.load(ply_file_path)

        # Normalize the mesh vertices
        normalize_mesh_vertices(mesh)

        # Sample and save complete and partial point clouds
        sample_and_save_points_as_xyz(mesh, complete_output_file_path, partial_output_file_path, num_complete_points,
                                      num_partial_points)

        # Create and save the occupancy grid
        create_and_save_occupancy_grid(mesh, occupancy_output_file_path, voxel_resolution)

        print(f"Processed {filename} - Complete and partial point clouds, and occupancy grid saved.")

print("All files processed!")

