import os
import shutil

# Define source and destination directories
source_dir = "/Users/cihankatar/Desktop/Github_Repo/Att-Next/PH2 Dataset images"
dermoscopic_images_dir = "dermoscopic_images"
lesion_images_dir = "lesion_images"

# Create destination directories if they don't exist
os.makedirs(dermoscopic_images_dir, exist_ok=True)
os.makedirs(lesion_images_dir, exist_ok=True)

# Iterate through each folder in the source directory
for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)
    
    # Check if the folder_path is a directory
    if os.path.isdir(folder_path):
        # Define paths to dermoscopic and lesion subfolders
        dermoscopic_path = os.path.join(folder_path, f"{folder_name}_Dermoscopic_Image")
        lesion_path = os.path.join(folder_path, f"{folder_name}_lesion")

        # Move dermoscopic images to the dermoscopic_images_dir
        if os.path.exists(dermoscopic_path):
            for file_name in os.listdir(dermoscopic_path):
                if file_name.endswith(".bmp"):
                    shutil.move(os.path.join(dermoscopic_path, file_name), dermoscopic_images_dir)

        # Move lesion images to the lesion_images_dir
        if os.path.exists(lesion_path):
            for file_name in os.listdir(lesion_path):
                if file_name.endswith(".bmp"):
                    shutil.move(os.path.join(lesion_path, file_name), lesion_images_dir)

print("Files have been successfully organized.")
