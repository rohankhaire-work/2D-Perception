import matplotlib.pyplot as plt
import torch


def compute_iou(pred_boxes, gt_boxes):
    """
    Compute IoU between predicted boxes (N,4) and ground truth boxes (M,4).

    Args:
        pred_boxes: (N, 4) tensor of predicted boxes in [xmin, ymin, xmax, ymax] format.
        gt_boxes: (M, 4) tensor of ground truth boxes in [xmin, ymin, xmax, ymax] format.

    Returns:
        IoU matrix (N, M) where each value is IoU between a predicted and a ground truth box.
    """
    # Ensure boxes are (N, 4) and (M, 4)
    assert pred_boxes.shape[1] == 4 and gt_boxes.shape[
        1] == 4, "Boxes must have shape (N,4) and (M,4)"

    # Expand dimensions to allow broadcasting (N,1,4) and (1,M,4)
    pred_boxes = pred_boxes.unsqueeze(1)  # (N, 1, 4)
    gt_boxes = gt_boxes.unsqueeze(0)  # (1, M, 4)

    # Extract coordinates
    x1_p, y1_p, x2_p, y2_p = pred_boxes[..., 0], pred_boxes[..., 1], \
        pred_boxes[..., 2], pred_boxes[..., 3]
    x1_g, y1_g, x2_g, y2_g = gt_boxes[..., 0], gt_boxes[..., 1], \
        gt_boxes[..., 2], gt_boxes[..., 3]

    # Compute intersection (clamp ensures no negative values)
    x1_i = torch.max(x1_p, x1_g)
    y1_i = torch.max(y1_p, y1_g)
    x2_i = torch.min(x2_p, x2_g)
    y2_i = torch.min(y2_p, y2_g)

    inter_width = (x2_i - x1_i)
    inter_height = (y2_i - y1_i)
    inter_area = inter_width * inter_height  # (N, M)

    # Compute union
    pred_area = (x2_p - x1_p) * \
        (y2_p - y1_p)  # (N,1)
    gt_area = (x2_g - x1_g) * (y2_g - y1_g)  # (1,M)
    union_area = pred_area + gt_area - inter_area  # (N, M)

    # Compute IoU (add epsilon to avoid division by zero)
    iou = inter_area / (union_area + 1e-6)
    return iou  # (N, M)


def clip_bboxes(bboxes, img_size):
    """
    Clips bounding boxes so they stay within image boundaries.

    Args:
        bboxes (Tensor): Bounding boxes of shape (B, N, 4) in (x1, y1, x2, y2) format.
        img_width (int): Width of the image.
        img_height (int): Height of the image.

    Returns:
        Tensor: Clipped bounding boxes.
    """
    bboxes_clone = bboxes.clone()
    img_width, img_height = img_size
    # Clip x1, x2 to [0, img_width]
    bboxes_clone[..., 0] = torch.clamp(
        bboxes[..., 0], min=0, max=img_width)  # x1
    bboxes_clone[..., 2] = torch.clamp(
        bboxes[..., 2], min=0, max=img_width)  # x2

    # Clip y1, y2 to [0, img_height]
    bboxes_clone[..., 1] = torch.clamp(
        bboxes[..., 1], min=0, max=img_height)  # y1
    bboxes_clone[..., 3] = torch.clamp(
        bboxes[..., 3], min=0, max=img_height)  # y2

    return bboxes_clone


def convert_pred_to_coco(pred_boxes: torch.Tensor, img_size: tuple) -> torch.Tensor:
    """
    Convert YOLO-NAS bounding boxes from (xc, yc, w, h) centered at (0,0)
    to (xmin, ymin, xmax, ymax) with top-left origin (0,0).

    Args:
        pred_boxes (torch.Tensor): (B, N, 4) tensor of (xc, yc, w, h) boxes.
        img_size (tuple): (width, height) of the image.

    Returns:
        torch.Tensor: (B, N, 4) tensor of (xmin, ymin, xmax, ymax) boxes.
    """
    img_width, img_height = img_size

    # Extract xc, yc, w, h
    xc, yc, w, h = pred_boxes[..., 0], pred_boxes[..., 1], \
        pred_boxes[..., 2], pred_boxes[..., 3]

    # Convert to (xmin, ymin, xmax, ymax) with top-left origin
    xmin = xc - (w / 2) + (img_width / 2)
    ymin = yc - (h / 2) + (img_height / 2)
    xmax = xc + (w / 2) + (img_width / 2)
    ymax = yc + (h / 2) + (img_height / 2)

    # Stack results back into (B, N, 4) shape
    converted_boxes = torch.stack((xmin, ymin, w, h), dim=-1)

    return converted_boxes


def normalize_xyxy(boxes: torch.Tensor, img_size: tuple) -> torch.Tensor:
    """
    Normalizes bounding boxes in (x_min, y_min, x_max, y_max) format 
    to be in the range [0, 1] relative to the given image size.

    Args:
        boxes (torch.Tensor): Tensor of shape (N, 4) with (x_min, y_min, x_max, y_max).
        img_size (tuple): (image_width, image_height).

    Returns:
        torch.Tensor: Normalized boxes with values in range [0, 1].
    """
    boxes_copy = boxes.clone()
    img_width, img_height = img_size  # Extract width and height

    # Create a tensor of shape (1, 4) for broadcasting
    img_size_tensor = torch.tensor([img_width, img_height, img_width, img_height],
                                   dtype=boxes.dtype, device=boxes.device)

    # Normalize boxes (broadcasting applies division to all rows)
    return boxes_copy / img_size_tensor


def normalize_boxes_xywh(boxes: torch.Tensor, img_size: tuple) -> torch.Tensor:
    """
    Normalizes bounding boxes from absolute pixel values to [0, 1] scale.

    Args:
        boxes (torch.Tensor): Tensor of shape (B, N, 4) with [xmin, ymin, xmax, ymax].
        img_width (int): Image width.
        img_height (int): Image height.

    Returns:
        torch.Tensor: Normalized tensor of shape (B, N, 4).
    """
    img_width, img_height = img_size
    # Clone to avoid modifying original tensor
    norm_boxes = boxes.clone()

    # Normalize x coordinates by image width
    norm_boxes[:, :, [0, 2]] /= img_width   # xmin, xmax

    # Normalize y coordinates by image height
    norm_boxes[:, :, [1, 3]] /= img_height  # ymin, ymax

    return norm_boxes


def xyxy_to_xywh(boxes):
    """
    Converts a tensor of bounding boxes from (x_min, y_min, x_max, y_max) 
    to (x_center, y_center, width, height) format.

    Args:
        boxes (torch.Tensor): Shape (N, 4) tensor with (x_min, y_min, x_max, y_max)

    Returns:
        torch.Tensor: Shape (N, 4) tensor with (x_center, y_center, width, height)
    """
    x_min, y_min, x_max, y_max = boxes[:,
                                       0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min

    return torch.stack([x_center, y_center, w, h], dim=1)


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


def plot_learning_curve(train_losses, valid_losses, valid_accuracies):
    plt.figure(figsize=(12, 6))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(valid_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(valid_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.show()


def log_loss(epoch, batch, total_loss, ciou_loss, l1_iou_loss, obj_loss, cls_loss, log_file):
    with open(log_file, "a") as f:  # Append mode
        f.write(f"Epoch: {epoch}, Batch: {batch}, Total Loss: {total_loss:.4f}, "
                f"CIOU Loss: {ciou_loss:.4f}, L1+IOU Loss: {l1_iou_loss:.4f}, "
                f"Object Loss: {obj_loss:.4f}, Class Loss: {cls_loss:.4f}\n")
