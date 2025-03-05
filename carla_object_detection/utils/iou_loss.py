from utils.utility_func import compute_iou
import torch


def iou_loss(pred_boxes, gt_boxes):
    """IoU loss calculation between predicted and ground truth boxes."""

    # Unpack coordinates
    x1_p, y1_p, x2_p, y2_p = pred_boxes[:, 0:1], \
        pred_boxes[:, 1:2], pred_boxes[:, 2:3], pred_boxes[:, 3:4]
    x1_g, y1_g, x2_g, y2_g = gt_boxes[:, 0], gt_boxes[:, 1], \
        gt_boxes[:, 2], gt_boxes[:, 3]

    # Compute Intersection
    x1_i = torch.max(x1_p, x1_g)  # (N, M)
    y1_i = torch.max(y1_p, y1_g)
    x2_i = torch.min(x2_p, x2_g)
    y2_i = torch.min(y2_p, y2_g)

    inter_area = (x2_i - x1_i).clamp(0) * \
        (y2_i - y1_i).clamp(0)  # Intersection area
    pred_area = (x2_p - x1_p) * (y2_p - y1_p)  # Predicted box area
    gt_area = (x2_g - x1_g) * (y2_g - y1_g)  # Ground truth box area

    # Compute IoU
    union_area = pred_area + gt_area - inter_area
    iou = inter_area / (union_area + 1e-6)  # Avoid division by zero

    # IoU
    return 1 - iou  # Loss = 1 - iou_loss


def ciou_loss(pred_boxes, gt_boxes):
    """
    Compute the Complete IoU (CIoU) between predicted (N,4) and ground truth (M,4) boxes.

    Args:
        pred_boxes (Tensor): Predicted bounding boxes (N, 4) in format [x, y, w, h]
        gt_boxes (Tensor): Ground truth bounding boxes (M, 4) in format [x, y, w, h]

    Returns:
        Tensor: CIoU loss (N, M)
    """

    iou_threshold = 0.3
    with torch.no_grad():
        iou = compute_iou(pred_boxes, gt_boxes)
    # Get the best prediction index for each GT box
    best_iou, best_gt_idx = iou.max(dim=1)  # (M,)

    # Mask for matched predictions (IoU above threshold)
    matched_mask = best_iou > iou_threshold
    matched_pred_boxes = pred_boxes[matched_mask]  # Select matched predictions
    # matched_gt_boxes = gt_boxes[best_gt_idx[matched_mask]]  # Corresponding GTs

    # print(matched_pred_boxes)
    # print(matched_mask)
    # print(best_gt_idx)

    # Compute CIoU loss for matched predictions
    if matched_pred_boxes.shape[0] > 0:
        ciou_loss_value = compute_ciou_loss(
            matched_pred_boxes, gt_boxes).mean()
    else:
        ciou_loss_value = torch.tensor(
            0.0, requires_grad=True)

    matched_gt_mask = iou.max(dim=0).values > iou_threshold
    missed_gt_loss = torch.tensor(0.0)
    if matched_gt_mask.sum() < gt_boxes.shape[0]:  # If some GTs are unmatched
        missed_gt_loss = torch.tensor(1.0, device=pred_boxes.device)

    # Combine all losses
    total_loss = ciou_loss_value + missed_gt_loss
    return total_loss, matched_mask


def compute_ciou_loss(pred_boxes, gt_boxes):
    """
    Compute the Complete IoU (CIoU) loss between predicted (N,4) and ground truth (M,4) boxes.

    Args:
        pred_boxes (Tensor): Predicted bounding boxes (N, 4) in [xmin, ymin, xmax, ymax] format.
        gt_boxes (Tensor): Ground truth bounding boxes (M, 4) in [xmin, ymin, xmax, ymax] format.

    Returns:
        Tensor: CIoU loss (N, M)
    """

    # Expand dimensions for broadcasting
    pred_boxes = pred_boxes[:, None, :]  # (N, 1, 4)
    gt_boxes = gt_boxes[None, :, :]      # (1, M, 4)

    # Unpack coordinates
    x1_p, y1_p, x2_p, y2_p = pred_boxes[..., 0], pred_boxes[..., 1], \
        pred_boxes[..., 2], pred_boxes[..., 3]
    x1_g, y1_g, x2_g, y2_g = gt_boxes[..., 0], gt_boxes[..., 1], \
        gt_boxes[..., 2], gt_boxes[..., 3]

    # Compute intersection
    x1_i = torch.max(x1_p, x1_g)
    y1_i = torch.max(y1_p, y1_g)
    x2_i = torch.min(x2_p, x2_g)
    y2_i = torch.min(y2_p, y2_g)

    inter_area = (x2_i - x1_i).clamp(0) * (y2_i - y1_i).clamp(0)
    pred_area = (x2_p - x1_p) * (y2_p - y1_p)
    gt_area = (x2_g - x1_g) * (y2_g - y1_g)

    # Compute IoU
    union_area = pred_area + gt_area - inter_area
    iou = inter_area / (union_area + 1e-6)

    # Compute box centers
    x_p_center, y_p_center = (x1_p + x2_p) / 2, (y1_p + y2_p) / 2
    x_g_center, y_g_center = (x1_g + x2_g) / 2, (y1_g + y2_g) / 2

    # Compute squared Euclidean distance between centers
    rho2 = (x_p_center - x_g_center) ** 2 + (y_p_center - y_g_center) ** 2

    # Compute enclosing box
    x1_c = torch.min(x1_p, x1_g)
    y1_c = torch.min(y1_p, y1_g)
    x2_c = torch.max(x2_p, x2_g)
    y2_c = torch.max(y2_p, y2_g)

    # Compute diagonal squared length of enclosing box
    c2 = (x2_c - x1_c) ** 2 + (y2_c - y1_c) ** 2

    # Aspect ratio penalty term
    w_p, h_p = (x2_p - x1_p).clamp(0), (y2_p - y1_p).clamp(0)
    w_g, h_g = (x2_g - x1_g).clamp(0), (y2_g - y1_g).clamp(0)

    v = (4 / (torch.pi ** 2)) * \
        ((torch.atan(w_g / (h_g + 1e-6)) - torch.atan(w_p / (h_p + 1e-6))) ** 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-6)

    # Compute CIoU
    ciou = iou - (rho2 / (c2 + 1e-6)) - (alpha * v)

    return 1 - ciou  # CIoU Loss (N, M)
