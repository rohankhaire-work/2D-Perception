import torch
from utils.utility_func import clip_bboxes, normalize_xyxy, \
    xywh_to_xyxy


def process_prediction(preds, img_size):
    # Get the model predictions from forward pass
    pred_bboxes, pred_scores = preds[0]
    cls_logits = preds[1][0]

    nms_top_k = 50
    batch_size, num_anchors, _ = pred_scores.size()

    pred_cls_conf, _ = torch.max(pred_scores, dim=2)  # [B, Anchors]
    topk_candidates = torch.topk(
        pred_cls_conf, dim=1, k=nms_top_k, largest=True, sorted=True)

    offsets = num_anchors * \
        torch.arange(batch_size, device=pred_cls_conf.device)
    indices_with_offset = topk_candidates.indices + \
        offsets.reshape(batch_size, 1)
    flat_indices = torch.flatten(indices_with_offset)

    output_pred_bboxes = pred_bboxes.reshape(-1, pred_bboxes.size(
        2))[flat_indices, :].reshape(pred_bboxes.size(0), nms_top_k, pred_bboxes.size(2))
    output_pred_scores = pred_scores.reshape(-1, pred_scores.size(
        2))[flat_indices, :].reshape(pred_scores.size(0), nms_top_k, pred_scores.size(2))
    output_cls_logits = cls_logits.reshape(-1, cls_logits.size(
        2))[flat_indices, :].reshape(cls_logits.size(0), nms_top_k, cls_logits.size(2))

    # Clip the bboxes and normalize the mfor CIOU loss
    output_pred_bboxes = clip_bboxes(output_pred_bboxes, img_size)
    output_pred_bboxes = normalize_xyxy(output_pred_bboxes, img_size)

    return output_pred_bboxes, output_pred_scores, output_cls_logits


def process_groundtruth(targets, img_size):
    bboxes = []
    labels = []

    # Iterate through the batch
    for target in targets:
        image_bboxes = [ann["bbox"]
                        for ann in target]
        image_labels = [ann["category_id"]
                        for ann in target]

        # Convert xywh to xyxy
        box_tensor = torch.tensor(image_bboxes, dtype=torch.float32)

        if box_tensor.shape[0] != 0:
            box_tensor = xywh_to_xyxy(box_tensor)
            box_tensor = clip_bboxes(box_tensor, img_size)
            den_tensor = torch.full((4,), img_size[0])
            final_tensor = box_tensor / den_tensor
        else:
            final_tensor = box_tensor

        bboxes.append(final_tensor)
        labels.append(torch.tensor(image_labels, dtype=torch.long))

    return bboxes, labels
