from torchvision.datasets import CocoDetection
import torchvision.transforms as transforms

IMG_X, IMG_Y = 512, 512
# Define image transform (resize and convert to tensor)
transform = transforms.Compose([
    transforms.Resize((IMG_X, IMG_Y)),  # Resize all images to 512x512
    transforms.ToTensor()
])


# Function to resize bounding boxes after image resizing
def resize_boxes(targets, orig_size, new_size):
    ratio_w = new_size[0] / orig_size[0]
    ratio_h = new_size[1] / orig_size[1]

    for ann in targets:
        ann["bbox"][0] *= ratio_w  # Scale x
        ann["bbox"][1] *= ratio_h  # Scale y
        ann["bbox"][2] *= ratio_w  # Scale width
        ann["bbox"][3] *= ratio_h  # Scale height
    return targets

# Custom COCO dataset class with bbox scaling


class CarlaObjects(CocoDetection):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)  # Load image & annotations

        # Get original image size before resizing
        orig_size = img.size  # (width, height)

        # Apply transformations (resize + convert to tensor)
        img = transform(img)

        # Resize bounding boxes to match new image size
        target = resize_boxes(target, orig_size, (IMG_X, IMG_Y))

        return img, target
