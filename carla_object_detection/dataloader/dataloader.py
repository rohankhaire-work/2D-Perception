from torchvision.datasets import CocoDetection


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
    def __init__(self, root, annFile, img_size, transform=None, target_transform=None):
        super().__init__(root, annFile)  # Initialize COCO dataset
        self.transform = transform  # Image transformations
        # Target (bbox) transformations
        self.target_transform = target_transform
        self.img_size = img_size

    def __getitem__(self, index):
        img, target = super().__getitem__(index)  # Load image & annotations

        orig_size = img.size  # (width, height) before transforms

        # Apply image transformations if specified
        if self.transform:
            img = self.transform(img)

        # Resize bounding boxes if needed
        if self.target_transform:
            target = resize_boxes(target, orig_size, self.img_size)

        return img, target
