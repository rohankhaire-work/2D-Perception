import torch
from torchvision.ops import box_iou, sigmoid_focal_loss


def compute_cls_loss(pred_boxes, gt_boxes, logits, gt_label, num_classes):

    device = logits.device
    if len(gt_label) == 0:
        loss = torch.tensor(0.0, requires_grad=True, device=device)
        return loss

    iou_thresh = 0.3
    # Compute iou
    iou_mat = box_iou(pred_boxes, gt_boxes)
    pred_idx, gt_idx = torch.where(iou_mat > iou_thresh)

    if len(pred_idx) == 0:
        loss = torch.tensor(0.0, requires_grad=True, device=device)
        return loss

    one_hot_array = []
    final_logits = logits[pred_idx]
    for i, j in zip(pred_idx, gt_idx):
        # Create one hot encoding vector
        class_label = gt_label[j]
        # There are 6 classes
        one_hot = torch.zeros(num_classes, dtype=torch.float32)
        one_hot[class_label] = 1

        one_hot_array.append(one_hot)

    # Convert to a single tensor
    one_hot_tensor = torch.stack(one_hot_array)
    one_hot_tensor = one_hot_tensor.to(device)

    loss = sigmoid_focal_loss(
        final_logits, one_hot_tensor, reduction="mean")

    return loss
