import os
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the directories
base_dir = "/home/haoming/Downloads/ycb_grasp_dataset"
input_dir = os.path.join(base_dir, "input")
gt_dir = os.path.join(base_dir, "gt")

# Function to load an XYZ file and return the points as a numpy array
def load_xyz(file_path):
    return np.loadtxt(file_path)

# Function to load an occupancy grid (npy format)
def load_occupancy_grid(file_path):
    return np.load(file_path)

# Function to visualize the point clouds and occupancy grid for 10 samples
def visualize_data(complete_points_list, partial_points_list, occupancy_grid_list):
    fig = plt.figure(figsize=(15, 10))

    # Loop through all 10 samples
    for i in range(10):
        ax1 = fig.add_subplot(3, 10, i + 1, projection='3d')
        ax1.scatter(complete_points_list[i][:, 0], complete_points_list[i][:, 1], complete_points_list[i][:, 2], c='b', s=1)
        ax1.set_title(f'Complete {i}')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        ax2 = fig.add_subplot(3, 10, i + 11, projection='3d')
        ax2.scatter(partial_points_list[i][:, 0], partial_points_list[i][:, 1], partial_points_list[i][:, 2], c='r', s=1)
        ax2.set_title(f'Partial {i}')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        ax3 = fig.add_subplot(3, 10, i + 21, projection='3d')
        ax3.voxels(occupancy_grid_list[i], edgecolor='k')
        ax3.set_title(f'Occupancy {i}')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')

    plt.tight_layout()
    plt.show()

# Function to randomly sample 10 corresponding partial, complete point clouds, and occupancy grids
def random_sample_visualization():
    # Get all the subfolders (categories) in the 'input' directory
    categories = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]

    # Randomly select a category (one of the 88 folders)
    selected_category = random.choice(categories)

    # Randomly choose "train" or "test"
    subset = random.choice(["train", "test"])

    # Get the paths for the chosen subset in input and gt folders
    partial_dir = os.path.join(input_dir, selected_category, subset)
    complete_dir = os.path.join(gt_dir, selected_category, subset)
    occupancy_dir = os.path.join(gt_dir, selected_category, subset)

    # Get all files (without the suffix) and randomly select 10 samples
    all_variants = [f.replace("_x.xyz", "") for f in os.listdir(partial_dir) if f.endswith("_x.xyz")]
    random_variants = random.sample(all_variants, 10)

    complete_points_list = []
    partial_points_list = []
    occupancy_grid_list = []

    # Load the corresponding partial, complete point clouds, and occupancy grids
    for variant in random_variants:
        partial_path = os.path.join(partial_dir, f"{variant}_x.xyz")
        complete_path = os.path.join(complete_dir, f"{variant}_y.xyz")
        occupancy_path = os.path.join(occupancy_dir, f"{variant}.npy")

        # Load the files
        partial_points = load_xyz(partial_path)
        complete_points = load_xyz(complete_path)
        occupancy_grid = load_occupancy_grid(occupancy_path)

        # Append to the respective lists
        partial_points_list.append(partial_points)
        complete_points_list.append(complete_points)
        occupancy_grid_list.append(occupancy_grid)

    # Visualize the 10 samples
    visualize_data(complete_points_list, partial_points_list, occupancy_grid_list)

# Run the visualization
if __name__ == "__main__":
    random_sample_visualization()




