from utils.iou_loss import xywh_to_xyxy
import torch
import torch.nn as nn
import torchvision.ops as ops


def compute_iou(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor):
    """
    Compute the IoU (Intersection over Union) between multiple predicted and ground truth bounding boxes.

    Args:
        pred_boxes (torch.Tensor): Tensor of shape (N, 4) containing predicted boxes in (x, y, w, h) format.
        gt_boxes (torch.Tensor): Tensor of shape (M, 4) containing ground truth boxes in (x, y, w, h) format.

    Returns:
        torch.Tensor: IoU matrix of shape (N, M) where IoU[i, j] is the IoU between pred_boxes[i] and gt_boxes[j].
    """

    device = pred_boxes.device

    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return torch.zeros((pred_boxes.shape[0], gt_boxes.shape[0]), device=pred_boxes.device)

    # Convert (x, y, w, h) to (x1, y1, x2, y2)
    pred_boxes_xyxy = xywh_to_xyxy(pred_boxes).to(device)
    gt_boxes_xyxy = xywh_to_xyxy(gt_boxes).to(device)

    # Compute IoU using torchvision's built-in function
    iou_matrix = ops.box_iou(
        pred_boxes_xyxy, gt_boxes_xyxy).to(device)  # (N, M)

    return iou_matrix


def obj_loss(pred_boxes, gt_boxes, pred_scores, gt_label):

    device = pred_boxes.device
    # IOU threshold
    iou_thresh = 0.2
    gt_label = gt_label.to(device)
    # Compute iou
    iou_mat = compute_iou(pred_boxes, gt_boxes)
    i, j = torch.where(iou_mat > iou_thresh)

    objective_arr = [0] * len(pred_boxes)

    for pred_idx, gt_idx in zip(i, j):
        # Create one hot encoding vector
        class_label = gt_label[gt_idx].to(device)
        multi_hot = torch.zeros(6, dtype=torch.float32).to(device)
        multi_hot[class_label] = 1

        if isinstance(objective_arr[pred_idx], torch.Tensor):
            continue

        objective_arr[pred_idx] = multi_hot

    for i, item in enumerate(objective_arr):
        if isinstance(item, torch.Tensor):
            continue
        objective_arr[i] = torch.zeros(6, dtype=torch.float32).to(device)

    # Convert to a single tensor
    # flattened_tensors = [torch.stack(sublist) for sublist in objective_arr]
    final_tensor = torch.stack(objective_arr)
    bce_loss = nn.BCEWithLogitsLoss()
    loss = bce_loss(pred_scores, final_tensor)

    return loss
