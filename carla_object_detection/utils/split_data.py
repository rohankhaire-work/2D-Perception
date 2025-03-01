import os
import shutil
import random

# Set paths
source_folder = "data/temp"  # Folder containing all images
output_folder = "data/carla_data"  # Destination folder

# Split ratios (adjust as needed)
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Create output directories
train_folder = os.path.join(output_folder, "train")
val_folder = os.path.join(output_folder, "val")
test_folder = os.path.join(output_folder, "test")

os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Get list of all images
images = [f for f in os.listdir(
    source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
random.shuffle(images)  # Shuffle for randomness

# Compute split indices
total_images = len(images)
train_count = int(total_images * train_ratio)
val_count = int(total_images * val_ratio)

# Split images
train_images = images[:train_count]
val_images = images[train_count:train_count + val_count]
test_images = images[train_count + val_count:]

# Move images to respective folders
for img in train_images:
    shutil.move(os.path.join(source_folder, img),
                os.path.join(train_folder, img))
    txt_file = os.path.splitext(img)[0] + ".txt"
    shutil.move(os.path.join(source_folder, txt_file),
                os.path.join(train_folder, txt_file))

for img in val_images:
    shutil.move(os.path.join(source_folder, img),
                os.path.join(val_folder, img))
    txt_file = os.path.splitext(img)[0] + ".txt"
    shutil.move(os.path.join(source_folder, txt_file),
                os.path.join(val_folder, txt_file))


for img in test_images:
    shutil.move(os.path.join(source_folder, img),
                os.path.join(test_folder, img))
    txt_file = os.path.splitext(img)[0] + ".txt"
    shutil.move(os.path.join(source_folder, txt_file),
                os.path.join(test_folder, txt_file))


print(
    f"✅ Split complete! Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
