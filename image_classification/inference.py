import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model.model import PlantDiseaseModel
from sklearn.preprocessing import LabelEncoder

import yaml
import os
import argparse

# Load config file
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

data_path = config["data"]["directory_root"]
# Get the number of labelels
labels = os.listdir(data_path)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

MODEL_DICT = {"resnet": models.resnet50(weights='IMAGENET1K_V1'),
              "mobilenet": models.mobilenet_v2(weights='IMAGENET1K_V1'),
              "custom": PlantDiseaseModel(len(labels))}

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True,
                help="name of the model")
ap.add_argument("--weight_path", required=True,
                help="path to trained model model")
ap.add_argument("--img_path", required=True,
                help="path to the image")
ap.add_argument("--use_cpu", action="store_true",
                help="whether to use cpu for inference")
args = vars(ap.parse_args())

# Load model
model = MODEL_DICT[args.model]

if args.use_cpu:
    device = torch.device("cpu")
    model.load_state_dict(torch.load(
        args.weight_path, map_location=device, weights_only=True))
else:
    device = torch.device("cuda")
    model.load_state_dict(torch.load(args.weight_path, weights_only=True))
    model.to(device)

model.eval()

# Preprocess input
transform = transforms.Compose([
    # Consistent resizing for validation/test
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                         0.229, 0.224, 0.225])
])

img = Image.open(args.img_path).convert("RGB")
img = transform(img)

if not args.use_cpu:
    img.to(device)

# Perform Inference
with torch.no_grad():
    output = model(img)
    _, predicted = torch.max(output, 1)

# Show image and label
plt.figure()
decoded_label = label_encoder.inverse_transform(predicted)
plt.title("Prediction: ", decoded_label)
plt.imshow(img.cpu().data)
