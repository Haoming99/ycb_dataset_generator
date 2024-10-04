import os

# Base directory paths
downloads_dir = "/home/haoming/Downloads"
ycb_grasp_dataset_dir = os.path.join(downloads_dir, "ycb_grasp_dataset")
gt_dir = os.path.join(ycb_grasp_dataset_dir, "gt")
input_dir = os.path.join(ycb_grasp_dataset_dir, "input")
partial_pcs_dir = "/home/haoming/Downloads/ycb_meshes/ycb-objects/partial_pcs"

# Create main directories
os.makedirs(gt_dir, exist_ok=True)
os.makedirs(input_dir, exist_ok=True)

# Get all partial point cloud files from the partial_pcs folder
partial_files = [f for f in os.listdir(partial_pcs_dir) if f.endswith(".xyz")]

# Create folders for each partial point cloud in both 'gt' and 'input'
for partial_file in partial_files:
    partial_name = partial_file.replace(".xyz", "")

    # Create train and test directories in gt and input for each partial point cloud
    gt_partial_dir = os.path.join(gt_dir, partial_name)
    input_partial_dir = os.path.join(input_dir, partial_name)

    os.makedirs(os.path.join(gt_partial_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(gt_partial_dir, "test"), exist_ok=True)
    os.makedirs(os.path.join(input_partial_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(input_partial_dir, "test"), exist_ok=True)

print("Folder structure created successfully!")