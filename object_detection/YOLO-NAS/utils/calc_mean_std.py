from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader.dataloader import CarlaObjects
import torch
from tqdm import tqdm
import yaml


# Custom collate function for variable-sized annotations
def collate_fn(batch):
    images, targets = zip(*batch)  # Unzip batch
    images = torch.stack(images, dim=0)  # Stack images to tensor
    # Keep targets as a list (they have different sizes)
    return images, targets


# Read the YAML file
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

train_path = config["data"]["train_dir"]
train_annotation = config["data"]["train_annotation"]
batch_size = config["training"]["batch_size"]
resize = config["data"]["augmentation"]["resize"]
img_size = (resize, resize)

# Define a transform to convert images to tensors
transform = transforms.ToTensor()

# Load the dataset (modify paths as needed)
# load coco dataset
train_dataset = CarlaObjects(root=train_path,
                             annFile=train_annotation,
                             img_size=img_size,
                             transform=transform)

# Use DataLoader to process images in batches
dataloader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4, collate_fn=collate_fn)
mean = torch.zeros(3)
std = torch.zeros(3)
total_images = 0

for images, _ in tqdm(dataloader, desc="Calculating Mean & Std"):
    batch_size = images.size(0)  # Number of images in the batch
    images = images.view(batch_size, 3, -1)  # Flatten each image to [3, H*W]

    mean += images.mean(dim=2).sum(dim=0)  # Compute mean per channel
    std += images.std(dim=2).sum(dim=0)  # Compute std deviation per channel

    total_images += batch_size

mean /= total_images
std /= total_images

print(f"Mean: {mean}")
print(f"Std Deviation: {std}")
