from utils.iou_loss import ciou_loss
from utils.cls_loss import compute_cls_loss
from utils.obj_loss import obj_loss


def compute_loss(pred_boxes, pred_scores, logits, gt_boxes, gt_labels,
                 epoch, batch_len, num_classes):
    batch_reg_loss = 0.0
    batch_cls_loss = 0.0
    batch_obj_loss = 0.0

    # Weights for CIOU and focal loss
    lambda_ciou = 1.25
    lambda_focal = 0.5
    lambda_obj = 2.5

    device = pred_boxes.device
    # Compute loss over the batch size
    for i in range(batch_len):

        # Pred boxes for this iteration
        pred_boxes_iter = pred_boxes[i]
        pred_scores_iter = pred_scores[i]
        pred_logits_iter = logits[i]

        # Transfer gt boxes and gt labels to device
        gt_boxes_iter = gt_boxes[i].to(device)
        gt_labels_iter = gt_labels[i].to(device)

        # Compute regression loss
        final_ciou_loss = ciou_loss(
            pred_boxes_iter, gt_boxes_iter)

        final_ciou_loss = final_ciou_loss.to(device)
        batch_reg_loss += final_ciou_loss

        # Compute objectiveness loss
        object_loss = obj_loss(pred_boxes_iter, gt_boxes_iter,
                               pred_scores_iter, gt_labels_iter,
                               num_classes)
        batch_obj_loss += object_loss

        # Compute classification loss
        cls_loss = compute_cls_loss(pred_boxes_iter, gt_boxes_iter,
                                    pred_logits_iter, gt_labels_iter,
                                    num_classes)
        batch_cls_loss += cls_loss

    total_loss = lambda_ciou * batch_reg_loss + lambda_focal * \
        batch_cls_loss + lambda_obj * batch_obj_loss

    return total_loss
