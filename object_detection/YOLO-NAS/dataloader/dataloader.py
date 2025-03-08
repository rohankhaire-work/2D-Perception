from pycocotools.coco import COCO
import torch
import cv2
import os


# Custom COCO dataset class with bbox scaling
class COCODataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)

        # Load image
        img_info = self.coco.imgs[image_id]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]

        # Extract bounding boxes & labels
        bboxes = []
        labels = []
        for ann in annotations:
            x_min, y_min, box_w, box_h = ann['bbox']
            bboxes.append([x_min, y_min, box_w, box_h])
            labels.append(ann['category_id'])

        # Apply transformations
        if self.transform:
            transformed = self.transform(
                image=img, bboxes=bboxes, labels=labels)
            img = transformed["image"]
            bboxes = transformed["bboxes"]
            labels = transformed["labels"]

        return img, torch.tensor(bboxes), torch.tensor(labels)
