import torch
import cv2
from super_gradients.training import models
from super_gradients.common.object_names import Models
from torchvision.ops import nms
import matplotlib.pyplot as plt
import argparse
import yaml


def _get_decoded_predictions_from_model_output(outputs):
    """
    Get the decoded predictions from the PPYoloE/YoloNAS output.
    Depending on the model regime (train/eval) the output format may differ so this method picks the right output.

    :param outputs: Model's forward() return value
    :return:        Tuple of (bboxes, scores) of shape [B, Anchors, 4], [B, Anchors, C]
    """
    if isinstance(outputs, tuple) and len(outputs) == 2:
        if torch.is_tensor(outputs[0]) and torch.is_tensor(outputs[1]) and outputs[0].shape[1] == outputs[1].shape[1] and outputs[0].shape[2] == 4:
            # This path happens when we are using traced model or ONNX model without postprocessing for inference.
            predictions = outputs
        else:
            # First is model predictions, second element of tuple is logits for loss computation
            predictions = outputs[0]
    else:
        raise ValueError(f"Unsupported output format: {outputs}")

    return predictions


def process_model_output(outputs, conf_threshold, nms_threshold):
    """

    :param outputs: Outputs of model's forward() method
    :return:        List of lists of tensors of shape [Ni, 6] where Ni is the number of detections in i-th image.
                    Format of each row is [x1, y1, x2, y2, confidence, class]
    """
    nms_result = []
    predictions = _get_decoded_predictions_from_model_output(outputs)

    for pred_bboxes, pred_scores in zip(*predictions):
        # Cast to float to avoid lack of fp16 support in torchvision.ops.boxes.batched_nms when doing CPU inference
        pred_bboxes = pred_bboxes.float()  # [Anchors, 4]
        pred_scores = pred_scores.float()  # [Anchors, C]

        # Filter all predictions by self.score_threshold
        i, j = (pred_scores > conf_threshold).nonzero(as_tuple=False).T
        pred_bboxes = pred_bboxes[i]
        pred_cls_conf = pred_scores[i, j]
        pred_cls_label = j[:]

        # NMS
        idx_to_keep = nms(pred_bboxes, pred_cls_conf,
                          iou_threshold=nms_threshold)

        pred_cls_conf = pred_cls_conf[idx_to_keep].unsqueeze(-1)
        pred_cls_label = pred_cls_label[idx_to_keep].unsqueeze(-1)
        pred_bboxes = pred_bboxes[idx_to_keep, :]

        #  nx6 (x1, y1, x2, y2, confidence, class) in pixel units
        final_boxes = torch.cat(
            [pred_bboxes, pred_cls_conf, pred_cls_label], dim=1)  # [N,6]

        nms_result.append(final_boxes)

        return nms_result


# === FUNCTION TO RUN INFERENCE ===
def run_inference(image_path, conf_threshold, nms_threshold, img_size):
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    # Resize to model input size
    image_resized = cv2.resize(image_rgb, img_size)
    image_tensor = torch.tensor(image_resized).permute(
        2, 0, 1).float().unsqueeze(0) / 255.0  # Normalize & add batch dim
    image_tensor = image_tensor.to(device)

    # Run inference
    with torch.no_grad():
        predictions = model(image_tensor)

    # Extract results
    results = process_model_output(predictions, conf_threshold, nms_threshold)

    # Draw bounding boxes on the image
    for result in results:
        if len(result) != 0:
            x1, y1, x2, y2 = map(int, result[:4])
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"Class {result[4]}: {result[5]:.2f}"
            cv2.putText(image_rgb, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the image
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.title(f"Inference: {image_path}")
    plt.show()


# === RUN INFERENCE ON MULTIPLE IMAGES ===
if __name__ == "__main__":
    # === PARSE COMMAND-LINE ARGUMENTS ===
    parser = argparse.ArgumentParser(
        description="YOLO NAS Object Detection Inference")
    parser.add_argument("image_paths", nargs="+", help="Paths to input images")
    args = parser.parse_args()

    # Read the YAML file
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    conf_threshold = config["inference"]["conf_threshold"]
    nms_threshold = config["inference"]["nms_threshold"]
    checkpoint_path = config["inference"]["checkpoint_path"]
    resize = config["inference"]["resize"]
    img_size = (resize, resize)
    num_classes = config["inference"]["num_classes"]

    # === LOAD THE TRAINED MODEL ===
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Change to M or L if needed
    model = models.get(Models.YOLO_NAS_S,
                       pretrained_weights=None, num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval().to(device)

    # perform Inference
    for image_path in args.image_paths:
        run_inference(image_path, conf_threshold, nms_threshold, img_size)
