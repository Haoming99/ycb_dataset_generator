import os

# Define the base directories
ycb_grasp_dataset_dir = "/home/haoming/Downloads/ycb_grasp_dataset"
input_dir = os.path.join(ycb_grasp_dataset_dir, "input")
gt_dir = os.path.join(ycb_grasp_dataset_dir, "gt")

# Function to get a list of folder names
def get_folder_names(base_dir):
    folder_names = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    return folder_names

# Get folder names from input and gt directories
def main():
    input_folder_names = get_folder_names(input_dir)
    gt_folder_names = get_folder_names(gt_dir)

    print("Input Folder Names:", input_folder_names)
    print("GT Folder Names:", gt_folder_names)
    print(len(gt_folder_names))

if __name__ == "__main__":
    main()
