import torch
import torch.optim as optim
from dataloader.dataloader import COCODataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.optim.lr_scheduler as lr_scheduler
from super_gradients.training import models
from executors.train import EarlyStopping, train_model, \
    evaluate_model
from utils.utility_func import plot_learning_curve

import yaml
import os


# Custom collate function for variable-sized annotations
def collate_fn(batch):
    images, bboxes, labels = zip(*batch)  # Unzip batch
    images = torch.stack(images, dim=0)  # Stack images to tensor
    # Keep targets as a list (they have different sizes)
    return images, bboxes, labels

# learning rate setup


def burnin_schedule(batch_count):
    if batch_count < 1000:
        factor = pow(batch_count / 1000, 4)
    elif batch_count < 28000:
        factor = 1.0
    elif batch_count < 31500:
        factor = 0.1
    else:
        factor = 0.01
    return factor


# Read the YAML file
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)
print(config)

# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up the loded params
use_wandb = config["training"]["use_wandb"]
train_path = config["data"]["train_dir"]
val_path = config["data"]["val_dir"]
test_path = config["data"]["test_dir"]
train_annotation = config["data"]["train_annotation"]
val_annotation = config["data"]["val_annotation"]
test_annotation = config["data"]["test_annotation"]
batch_size = config["training"]["batch_size"]
save_path = config["training"]["save_path"]

# Data params
num_classes = config["data"]["classes"]
resize = config["data"]["augmentation"]["resize"]
img_size = (resize, resize)

# training params
use_checkpoint = config["training"]["use_checkpoint"]
weight_decay = config["model"]["hyperparameters"]["optimizer"]["weight_decay"]
lr = config["model"]["hyperparameters"]["optimizer"]["learning_rate"]
patience = config["training"]["early_stopping"]["patience"]
n_epochs = config["training"]["epochs"]

# Transform
valid_transforms = A.Compose([
    A.Resize(height=resize, width=resize),
    A.Normalize(mean=(0.4711, 0.4657, 0.4667), std=(
        0.1597, 0.1556, 0.1601)),
    ToTensorV2()], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))  # Adjust bboxes


train_transforms = A.Compose([
    # Geometric transformations
    A.HorizontalFlip(p=0.5),      # Flip left-right
    # Flip up-down (useful for aerial/satellite data)
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.5),      # Rotate by 90 degrees
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                       rotate_limit=10, p=0.5),

    # Photometric transformations
    A.RandomBrightnessContrast(p=0.3),  # Adjust brightness & contrast
    # Change hue/saturation for color diversity
    A.HueSaturationValue(p=0.3),
    A.RGBShift(p=0.2),                  # Slight RGB channel shift
    A.Blur(blur_limit=3, p=0.2),        # Random blur to simulate motion blur
    A.GaussNoise(p=0.2),                # Add random noise

    # Bounding-box friendly transformations
    A.CLAHE(p=0.2),                     # Adaptive histogram equalization
    A.ToGray(p=0.1),                    # Convert some images to grayscale
    # Resize images to model input size
    A.Resize(resize, resize),
    A.Normalize(mean=(0.4711, 0.4657, 0.4667), std=(
        0.1597, 0.1556, 0.1601)),  # ImageNet normalization
    ToTensorV2()], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))  # Adjust bboxes

# load coco dataset
train_dataset = COCODataset(train_path,
                            train_annotation,
                            transform=train_transforms)

val_dataset = COCODataset(val_path,
                          val_annotation,
                          transform=valid_transforms)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size,
                        shuffle=False, collate_fn=collate_fn)

# Check if save path exists else create it
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Use YOLO NAS nano
model = models.get("yolo_nas_s", pretrained_weights=None,
                   num_classes=num_classes)
if use_checkpoint:
    state_dict = torch.load(use_checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)

model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)
early_stopping = EarlyStopping(patience=patience, min_delta=0.001,
                               save_path=save_path + "/best_model.pth")

# Start training
train_losses, val_losses = train_model(
    model, train_loader, val_loader, optimizer, scheduler,
    epochs=n_epochs, img_size=img_size, num_classes=num_classes,
    early_stopping=early_stopping, wandb_log=use_wandb)

# plot_learning_curve(train_losses, val_losses)
