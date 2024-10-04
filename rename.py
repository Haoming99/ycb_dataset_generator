import os
import re

# Define the base directories
ycb_grasp_dataset_dir = "/home/haoming/Downloads/ycb_grasp_dataset"
input_dir = os.path.join(ycb_grasp_dataset_dir, "input")
gt_dir = os.path.join(ycb_grasp_dataset_dir, "gt")

# Function to remove the prefix from file names
def remove_prefix_from_files(base_dir):
    for category in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category)
        if os.path.isdir(category_path):
            for subset in ['train', 'test']:
                subset_path = os.path.join(category_path, subset)
                if os.path.exists(subset_path):
                    for file in os.listdir(subset_path):
                        if file.endswith(".npy") or file.endswith(".xyz"):
                            # Use regex to capture the number and the suffix (if present)
                            match = re.search(r"_(\d+(_[xy])?)\.(npy|xyz)", file)
                            if match:
                                new_name = f"{match.group(1)}.{match.group(3)}"
                                old_path = os.path.join(subset_path, file)
                                new_path = os.path.join(subset_path, new_name)
                                os.rename(old_path, new_path)
                                print(f"Renamed: {old_path} -> {new_path}")

# Execute renaming for both input and gt folders
def main():
    remove_prefix_from_files(input_dir)
    remove_prefix_from_files(gt_dir)

if __name__ == "__main__":
    main()
