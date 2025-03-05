from utils.iou_loss import ciou_loss
from utils.cls_loss import FocalLoss
from utils.obj_loss import obj_loss


def compute_loss(pred_boxes, pred_scores, logits, gt_boxes, gt_labels, epoch):
    batch_reg_loss = 0.0
    batch_cls_loss = 0.0
    batch_obj_loss = 0.0

    # Weights for CIOU and focal loss
    lambda_ciou = 1.5
    lambda_focal = 1.2
    lambda_obj = 1.5
    # Initiate Focal loss
    focal_loss_fn = FocalLoss(gamma=1.0, alpha=0.25)

    device = pred_boxes.device
    # Compute loss over the batch size
    for i in range(len(pred_boxes)):
        # Compute CIOU loss
        if len(gt_labels[i]) == 0:
            continue

        gt_boxes_iter = gt_boxes[i].to(device)
        final_ciou_loss, matched_mask = ciou_loss(pred_boxes[i], gt_boxes_iter)

        final_ciou_loss = final_ciou_loss.to(device)
        batch_reg_loss += final_ciou_loss

        useful_pred_boxes = pred_boxes[i][matched_mask]

        if useful_pred_boxes.size(0) != 0:
            # Compute objectiveness loss
            object_loss = obj_loss(pred_boxes[i][matched_mask], gt_boxes_iter,
                                   pred_scores[i][matched_mask], gt_labels[i])

            batch_obj_loss += object_loss
            # Compute classification loss
            cls_loss = focal_loss_fn(logits[i][matched_mask], gt_labels[i])
            batch_cls_loss += cls_loss

    total_loss = lambda_ciou * batch_reg_loss + lambda_focal * \
        batch_cls_loss + lambda_obj * batch_obj_loss

    return total_loss
