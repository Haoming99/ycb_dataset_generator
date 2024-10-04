import os

# Define the base directory
ycb_grasp_dataset_dir = "/home/haoming/Downloads/ycb_grasp_dataset"

# Function to delete .npy and .xyz files
def delete_npy_xyz_files():
    for root, dirs, files in os.walk(ycb_grasp_dataset_dir):
        for file in files:
            if file.endswith(".npy") or file.endswith(".xyz"):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

# Execute deletion
if __name__ == "__main__":
    delete_npy_xyz_files()
