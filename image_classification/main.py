from dataloader.data_loader import PlantDiseaseDataset, load_images
from model.model import PlantDiseaseModel
from utils.util import plot_learning_curve
from executors.train import train_model, evaluate_model, EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
import os
import yaml
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read the YAML file
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

print(config)

if config["training"]["use_wandb"]:
    exp_name = config["model"]["dataset"] + "_" + config["model"]["name"]
    wandb.init(project=exp_name)
    wandb.config = config

# Load data
image_paths, labels = load_images(config["data"]["directory_root"])

# Encode labels as integers
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Train, validation, and test splits
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths, labels_encoded, test_size=0.3,
    random_state=42, stratify=labels_encoded)
valid_paths, test_paths, valid_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.5,
    random_state=42, stratify=temp_labels)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                         0.229, 0.224, 0.225])  # Standard normalization
])

valid_test_transform = transforms.Compose([
    # Consistent resizing for validation/test
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                         0.229, 0.224, 0.225])
])

# Create datasets with appropriate transformations
train_dataset = PlantDiseaseDataset(
    train_paths, train_labels, transform=train_transform)
valid_dataset = PlantDiseaseDataset(
    valid_paths, valid_labels, transform=valid_test_transform)
test_dataset = PlantDiseaseDataset(
    test_paths, test_labels, transform=valid_test_transform)

# Create dataloaders
batch_size = config["training"]["batch_size"]
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)  # Shuffle for training
# No shuffle for validation/test
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=False)   # No shuffle for test

# Store parameters from config file
use_wandb = config["training"]["use_wandb"]
lr = config["model"]["hyperparameters"]["optimizer"]["learning_rate"]
momentum = config["model"]["hyperparameters"]["optimizer"]["momentum"]
num_classes = len(label_encoder.classes_)
n_epochs = config["training"]["epochs"]
patience = config["training"]["early_stopping"]["patience"]
save_path = config["training"]["save_path"]

# Check if save path exists else create it
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Training models
if config["transfer_learning"]["resnet"]:

    # Use RESNET50
    model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    early_stopping = EarlyStopping(patience=patience, min_delta=0.01,
                                   save_path=save_path + "/best_model.pth")

    # Start training
    train_losses, valid_losses, valid_accuracies = train_model(
        model, train_loader, valid_loader, criterion, optimizer,
        epochs=n_epochs, early_stopping=early_stopping, wandb_log=use_wandb)

    plot_learning_curve(train_losses, valid_losses, valid_accuracies)

    # Evaluate model on test data
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
    print("Test loss: ", test_loss)
    print("test accuracy: ", test_accuracy)

if config["transfer_learning"]["mobilenet"]:
    # Uses MobileNetV2
    model = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    early_stopping = EarlyStopping(patience=patience, min_delta=0.01,
                                   save_path=save_path + "/best_model.pth")

    # Training
    train_losses, valid_losses, valid_accuracies = train_model(
        model, train_loader, valid_loader, criterion, optimizer,
        epochs=n_epochs, early_stopping=early_stopping, wandb_log=use_wandb)

    plot_learning_curve(train_losses, valid_losses, valid_accuracies)

    # Evaluate model on test data
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
    print("Test loss: ", test_loss)
    print("test accuracy: ", test_accuracy)

if config["transfer_learning"]["custom"]:
    # Uses custom CNN
    model = PlantDiseaseModel(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    early_stopping = EarlyStopping(
        patience=4, min_delta=0.01, save_path=save_path + "/best_model.pth")
    train_losses, valid_losses, valid_accuracies = train_model(
        model, train_loader, valid_loader, criterion, optimizer,
        epochs=n_epochs, early_stopping=early_stopping, wandb_log=use_wandb)

    plot_learning_curve(train_losses, valid_losses, valid_accuracies)

    # Evaluate model on test data
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
    print("Test loss: ", test_loss)
    print("test accuracy: ", test_accuracy)
