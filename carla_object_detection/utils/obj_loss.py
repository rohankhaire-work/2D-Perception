from utils.utility_func import xywh_to_xyxy
import torch
import torch.nn as nn
import torchvision.ops as ops
from utils.utility_func import compute_iou


def obj_loss(pred_boxes, gt_boxes, pred_scores, gt_label):

    device = pred_boxes.device
    # IOU threshold
    iou_thresh = 0.2
    # gt_label = gt_label.to(device)
    # Compute iou
    iou_mat = compute_iou(pred_boxes, gt_boxes)
    i, j = torch.where(iou_mat > iou_thresh)

    objective_arr = [0] * len(pred_boxes)

    for pred_idx, gt_idx in zip(i, j):
        # Create one hot encoding vector
        class_label = gt_label[gt_idx]
        multi_hot = torch.zeros(6, dtype=torch.float32)
        multi_hot[class_label] = 1

        if isinstance(objective_arr[pred_idx], torch.Tensor):
            continue

        objective_arr[pred_idx] = multi_hot

    for i, item in enumerate(objective_arr):
        if isinstance(item, torch.Tensor):
            continue
        objective_arr[i] = torch.zeros(6, dtype=torch.float32)

    # Convert to a single tensor
    final_tensor = torch.stack(objective_arr)
    final_tensor = final_tensor.to(device)
    bce_loss = nn.BCELoss()
    loss = bce_loss(pred_scores, final_tensor)

    return loss
