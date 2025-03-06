import torch
from torchvision.ops import box_iou


def obj_loss(pred_boxes, gt_boxes, pred_scores, gt_label):

    device = pred_scores.device
    # IOU threshold
    iou_thresh = 0.3

    if len(gt_label) == 0:
        pred_idx = []
        gt_idx = []
    else:
        iou_mat = box_iou(pred_boxes, gt_boxes)
        pred_idx, gt_idx = torch.where(iou_mat > iou_thresh)

    one_hot_tensor = torch.zeros(6)
    final_pred_scores = torch.zeros(6)
    # If there are no overlapping pred pred boxes
    # penalize the conf scores of topk boxes
    if len(pred_idx) == 0:
        final_pred_scores = pred_scores
        one_hot_tensor = torch.zeros((len(pred_boxes), 6), dtype=torch.float32)
        one_hot_tensor = one_hot_tensor.to(device)
    else:
        one_hot_array = []
        final_pred_scores = pred_scores[pred_idx]
        for i, j in zip(pred_idx, gt_idx):
            # Create one hot encoding vector
            class_label = gt_label[j]
            # There are 6 classes
            one_hot = torch.zeros(6, dtype=torch.float32)
            one_hot[class_label] = 1

            one_hot_array.append(one_hot)

        # Convert to a single tensor
        one_hot_tensor = torch.stack(one_hot_array)
        one_hot_tensor = one_hot_tensor.to(device)

    loss = torch.nn.functional.binary_cross_entropy(
        final_pred_scores, one_hot_tensor)

    return loss
