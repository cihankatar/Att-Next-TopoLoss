import os
import shutil

# Define the source folder and destination folders
source_folder = '/Users/cihankatar/Desktop/Github_Repo/Att-Next/ISBI2016_ISIC_Part3B_Training_Data'
segmentation_folder = '/Users/cihankatar/Desktop/Github_Repo/Att-Next/Segmentation_Maps'
images_folder = '/Users/cihankatar/Desktop/Github_Repo/Att-Next/Normal_Images'

# Create destination folders if they don't exist
os.makedirs(segmentation_folder, exist_ok=True)
os.makedirs(images_folder, exist_ok=True)

# Iterate through the files in the source folder
for file_name in os.listdir(source_folder):
    # Full path to the file
    file_path = os.path.join(source_folder, file_name)
    
    # Check if it's a segmentation map or normal image
    if file_name.endswith('_Segmentation.png'):
        # Move to segmentation folder
        shutil.move(file_path, os.path.join(segmentation_folder, file_name))
        print(f"Moved {file_name} to Segmentation Maps folder")
    elif file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png'):
        # Move to normal images folder
        shutil.move(file_path, os.path.join(images_folder, file_name))
        print(f"Moved {file_name} to Normal Images folder")

print("All files have been organized.")
