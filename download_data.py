import kagglehub
import os
import shutil

# Download latest version
print("Downloading CT scan dataset from Kaggle...")
path = kagglehub.dataset_download("sampathlonka86/chestctscans")

print("Path to dataset files:", path)

# Copy to project data directory
project_data_dir = "/home/sgram/CTXray/data/ct_scans"
if os.path.exists(project_data_dir):
    print(f"Data directory already exists at {project_data_dir}")
else:
    print(f"Copying data to {project_data_dir}...")
    shutil.copytree(path, project_data_dir)
    print("Data copied successfully!")

print("\nExploring dataset structure...")
for root, dirs, files in os.walk(project_data_dir):
    level = root.replace(project_data_dir, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files[:5]:  # Show first 5 files
        print(f'{subindent}{file}')
    if len(files) > 5:
        print(f'{subindent}... and {len(files) - 5} more files')
