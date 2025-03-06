from utils.utility_func import compute_iou
import torch
from torchvision.ops import complete_box_iou_loss, distance_box_iou_loss, box_iou


def iou_loss(pred_boxes, gt_boxes):
    """IoU loss calculation between predicted and ground truth boxes."""

    iou = compute_iou(pred_boxes, gt_boxes)

    return 1 - iou.mean()  # Loss = 1 - iou_loss


def ciou_loss(pred_boxes, gt_boxes):
    """
    Compute the Complete IoU (CIoU) between predicted (N,4) and ground truth (M,4) boxes.

    Args:
        pred_boxes (Tensor): Predicted bounding boxes (N, 4) in format [x, y, w, h]
        gt_boxes (Tensor): Ground truth bounding boxes (M, 4) in format [x, y, w, h]

    Returns:
        Tensor: CIoU loss (N, M)
    """
    if (gt_boxes.size(0) == 0):
        loss = torch.tensor(0.0, requires_grad=True)
        return loss

    iou_threshold = 0.3

    iou = box_iou(pred_boxes, gt_boxes)
    # Get the best prediction index for each GT box
    best_iou, best_gt_idx = iou.max(dim=1)  # (M,)

    # Mask for matched predictions (IoU above threshold)
    matched_mask = best_iou > iou_threshold
    matched_pred_boxes = pred_boxes[matched_mask]  # Select matched predictions
    matched_gt_boxes = gt_boxes[best_gt_idx[matched_mask]]  # Corresponding GTs

    # Compute CIoU loss for matched predictions
    if matched_pred_boxes.shape[0] > 0:
        ciou_loss_value = complete_box_iou_loss(
            matched_pred_boxes, matched_gt_boxes, reduction="mean")

        # print("CIOUD LOSS:", ciou_loss_value)
    else:
        # If there are no over laps the nwe apply IOU loss

        l1_distances = torch.cdist(
            pred_boxes, gt_boxes, p=1)  # (N, M) L1 distances
        best_l1_idx = torch.argmin(l1_distances, dim=1)  # (N,)
        matched_gt_boxes = gt_boxes[best_l1_idx]
        final_iou_loss = (1 - iou.mean())
        # print("IOU LOSS: ", final_iou_loss)
        final_l1_loss = torch.nn.functional.l1_loss(
            pred_boxes, matched_gt_boxes).mean()
        # print("L1 LOSS: ", final_l1_loss)
        ciou_loss_value = final_l1_loss + final_iou_loss

    return ciou_loss_value
