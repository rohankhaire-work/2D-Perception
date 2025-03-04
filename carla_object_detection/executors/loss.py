from utils.iou_loss import ciou_loss
from utils.cls_loss import FocalLoss
import torch


def compute_loss(pred_boxes, logits, gt_boxes, gt_labels):
    batch_reg_loss = 0.0
    batch_cls_loss = 0.0

    # Weights for CIOU and focal loss
    lambda_ciou = 1.0
    lambda_focal = 1.0

    # Initiate Focal loss
    focal_loss_fn = FocalLoss(gamma=1.0, alpha=0.25)

    # Compute loss over the batch size
    for i in range(len(pred_boxes)):
        # Compute CIOU loss
        if len(gt_boxes[i]) == 0:
            continue

        topk = len(gt_labels[i])
        ciou_loss_mat = ciou_loss(pred_boxes[i][:topk], gt_boxes[i])
        matched_losses = torch.min(ciou_loss_mat, dim=1).values
        final_ciou_loss = matched_losses.mean()
        batch_reg_loss += final_ciou_loss

        # Compute classification loss
        cls_loss = focal_loss_fn(logits[i][:topk], gt_labels[i])
        batch_cls_loss += cls_loss

    total_loss = lambda_ciou * batch_reg_loss + lambda_focal * batch_cls_loss

    return total_loss
