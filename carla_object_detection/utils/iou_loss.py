import torch


def xywh_to_xyxy(boxes):
    """
    Convert bounding boxes from (x, y, w, h) format to (x1, y1, x2, y2).

    Args:
        boxes (Tensor): Tensor of shape (N, 4) or (M, 4) with [x, y, w, h] format.

    Returns:
        Tensor: Converted tensor with shape (N, 4) or (M, 4) in [x1, y1, x2, y2] format.
    """
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return torch.stack([x, y, x + w, y + h], dim=1)


def ciou_loss(pred_boxes, gt_boxes):
    """
    Compute the Complete IoU (CIoU) between predicted (N,4) and ground truth (M,4) boxes.

    Args:
        pred_boxes (Tensor): Predicted bounding boxes (N, 4) in format [x, y, w, h]
        gt_boxes (Tensor): Ground truth bounding boxes (M, 4) in format [x, y, w, h]

    Returns:
        Tensor: CIoU loss (N, M)
    """

    device = pred_boxes.device
    # Convert (x, y, w, h) → (x1, y1, x2, y2)
    pred_boxes = xywh_to_xyxy(pred_boxes).to(device)
    gt_boxes = xywh_to_xyxy(gt_boxes).to(device)

    N, M = pred_boxes.shape[0], gt_boxes.shape[0]

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
    w_p, h_p = x2_p - x1_p, y2_p - y1_p
    w_g, h_g = x2_g - x1_g, y2_g - y1_g

    v = (4 / (torch.pi ** 2)) * \
        ((torch.atan(w_g / h_g) - torch.atan(w_p / h_p)) ** 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-6)  # Dynamic trade-off parameter

    # Compute CIoU
    ciou = iou - (rho2 / (c2 + 1e-6)) - (alpha * v)

    return 1 - ciou  # CIoU Loss (higher loss for worse boxes)
