from dataloader.dataset import KittiDisparity, load_filenames
from dataloader.data_loader import input_rgb_image_loader, input_sgm, \
    train_target_gt_depth_loader, test_target_gt_depth_loader
from dataloader.transforms import ToTensor
from dataloader.co_transforms import Compose, RandomColorJitter, \
    RandomHorizontalFlip
from model.DilNet import DilNetLRDisp
from executors.train import train_network, EarlyStopping

import torch
import os
import yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read the YAML file
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

print(config)

# Store parameters from config file
lr = config["model"]["hyperparameters"]["optimizer"]["learning_rate"]
momentum = config["model"]["hyperparameters"]["optimizer"]["momentum"]
epochs = config["training"]["epochs"]
batch_size = config["training"]["batch_size"]
patience = config["training"]["early_stopping"]["patience"]
save_path = config["training"]["save_path"]
milestones = config["training"]["milestones"]
datadir = config["data"]["directory_root"]
train_input_files = config["data"]["train_input_files"]
train_target_files = config["data"]["train_target_files"]
test_input_files = config["data"]["test_input_files"]
test_target_files = config["data"]["test_target_files"]

# Load data
# Train files: Whole KITTI 2015 / Test files: Whole KITTI 2012
train_filenames = load_filenames(
    path_data=train_input_files,
    path_target=train_target_files,
    data_dir=datadir)

test_filenames = load_filenames(
    path_data=test_input_files,
    path_target=test_target_files,
    data_dir=datadir)

# Image loader and transforms
im_loader = [input_sgm, input_rgb_image_loader, input_rgb_image_loader]
train_co_transforms = Compose([RandomColorJitter(
    0.5, 0.5, 0.5, 0.35, 0.5), RandomHorizontalFlip()])
input_transforms = [ToTensor(torch.FloatTensor), ToTensor(
    torch.FloatTensor), ToTensor(torch.FloatTensor)]
target_transforms = [ToTensor(
    torch.FloatTensor), ToTensor(torch.LongTensor)]

train_dataset = KittiDisparity(
    filelist=train_filenames,
    image_loader=im_loader,
    target_loader=[train_target_gt_depth_loader],
    training=True,
    co_transforms=train_co_transforms,
    input_transforms=input_transforms,
    target_transforms=target_transforms,)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_dataset = KittiDisparity(
    filelist=test_filenames,
    image_loader=im_loader,
    target_loader=[test_target_gt_depth_loader],
    training=False,
    co_transforms=None,
    input_transforms=input_transforms,
    target_transforms=target_transforms,
    return_filenames=False)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=True, num_workers=0)

# Create model
net_refine = DilNetLRDisp(7, 1).to(device)

optimizer = torch.optim.Adam(net_refine.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=milestones, gamma=0.5)

# Early stopping
early_stopping = EarlyStopping(patience=patience, min_delta=0.01,
                               save_path=save_path + "/best_model.pth")
#################
# Train Loop    #
#################

# Check if save path exists else create it
if not os.path.exists(save_path):
    os.makedirs(save_path)

train_network(net_refine, optimizer, scheduler, epochs,
              train_loader, test_loader, early_stopping)
