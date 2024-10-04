import trimesh
import numpy as np
import os



def generate_and_save_occupancy_grid(ply_file_path, output_path, voxel_resolution):
    # Load the mesh from the PLY file
    mesh = trimesh.load(ply_file_path)

    # Voxelize the mesh
    voxelized_mesh = mesh.voxelized(pitch=mesh.scale / voxel_resolution)

    # Create an occupancy grid (as a numpy array)
    occupancy_grid = voxelized_mesh.matrix.astype(int)

    # Save the occupancy grid as a .npy file
    np.save(output_path, occupancy_grid)


if __name__ == "__main__":
    mesh_directory = "/home/haoming/Downloads/ycb_meshes/ycb-objects/meshes"
    output_directory = "/home/haoming/Downloads/ycb_meshes/ycb-objects/occupancy_grids"
    os.makedirs(output_directory, exist_ok=True)

    voxel_resolution = 32

    for filename in os.listdir(mesh_directory):
        if filename.endswith(".ply"):
            # Full path to the .ply file
            ply_file_path = os.path.join(mesh_directory, filename)

            # Output file name (same as the PLY file but with .npy extension)
            output_file_path = os.path.join(output_directory, filename.replace(".ply", ".npy"))

            # Generate and save occupancy grid
            generate_and_save_occupancy_grid(ply_file_path, output_file_path, voxel_resolution)

            print(f"Processed {filename} and saved occupancy grid to {output_file_path}")

    print("All files processed!")


