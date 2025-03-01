import torch
import torch.optim as optim
from dataloader.dataloader import CarlaObjects
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from super_gradients.training import models
from executors.train import EarlyStopping, train_model, \
    evaluate_model, YOLONASLoss
from utils.plot import plot_learning_curve

import yaml
import os


# Custom collate function for variable-sized annotations
def collate_fn(batch):
    images, targets = zip(*batch)  # Unzip batch
    images = torch.stack(images, dim=0)  # Stack images to tensor
    # Keep targets as a list (they have different sizes)
    return images, targets


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
resize = config["data"]["augmentation"]["resize"]

# training params
weight_decay = config["model"]["hyperparameters"]["optimizer"]["weight_decay"]
lr = config["model"]["hyperparameters"]["optimizer"]["learning_rate"]
patience = config["training"]["early_stopping"]["patience"]
n_epochs = config["training"]["epochs"]

# load coco dataset
train_dataset = CarlaObjects(root=train_path,
                             annFile=train_annotation)

val_dataset = CarlaObjects(root=val_path,
                           annFile=val_annotation)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size,
                        shuffle=False, collate_fn=collate_fn)

# Check if save path exists else create it
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Use YOLO NAS nano
model = models.get("yolo_nas_s", pretrained_weights=None, num_classes=6)
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

early_stopping = EarlyStopping(patience=patience, min_delta=0.001,
                               save_path=save_path + "/best_model.pth")

# Start training
train_losses, val_losses, valid_maps = train_model(
    model, train_loader, val_loader, optimizer,
    epochs=n_epochs, early_stopping=early_stopping, wandb_log=use_wandb)

plot_learning_curve(train_losses, val_losses, valid_maps)

# Evaluate model on test data
test_dataset = CarlaObjects(root=test_path,
                            annFile=test_annotation)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# Initiate loss function
loss_function = YOLONASLoss()
metric = MeanAveragePrecision(iou_thresholds=[0.5, 0.75])
# Test the model
test_loss, test_map = evaluate_model(model, test_loader, loss_function, metric)

print("Test loss: ", test_loss)
print("test accuracy: ", test_map)
