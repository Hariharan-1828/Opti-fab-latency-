import shutil
import os

source_image = "clean.jpg"      # image to duplicate
destination_folder = "clean"    # where copies go
num_copies = 120                 # number of copies

os.makedirs(destination_folder, exist_ok=True)

for i in range(1, num_copies + 1):
    new_name = f"Clean_{i}.jpg"
    shutil.copy(source_image, os.path.join(destination_folder, new_name))

print("Done! 100 images created.")
