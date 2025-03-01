import torch
import torch.nn as nn
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0, save_path="best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Save the best model
            torch.save(model.state_dict(), self.save_path)
            print(f"[INFO] Model checkpoint saved to {self.save_path}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("[INFO] Early stopping triggered.")
                return True
        return False


class YOLONASLoss(nn.Module):
    def __init__(self, bbox_weight=5.0, cls_weight=1.0):
        super(YOLONASLoss, self).__init__()
        self.bbox_loss = nn.MSELoss()  # Bounding box regression loss
        self.cls_loss = nn.CrossEntropyLoss()  # Classification loss
        self.bbox_weight = bbox_weight
        self.cls_weight = cls_weight

    def forward(self, outputs, targets):
        """
        Compute YOLO-NAS loss for the entire batch
        :param outputs: Tuple (pred_boxes, pred_scores)
        :param targets: List of lists, where each list contains multiple dicts
                        [{"bboxes": Tensor, "labels": Tensor}, ...] for each image.
        :return: Total batch loss (bbox loss + class loss)
        """
        pred_boxes, pred_scores = outputs  # Unpack model predictions

        bbox_loss_total = 0.0
        cls_loss_total = 0.0
        batch_size = len(targets)

        for i in range(batch_size):
            image_targets = targets[i]  # List of dicts for this image

            if not isinstance(image_targets, list) or len(image_targets) == 0:
                continue  # Skip if there are no objects in this image

            # Collect all bounding boxes and labels for this image
            gt_bboxes = []
            gt_labels = []

            for obj in image_targets:
                if "bbox" in obj and "category_id" in obj:
                    gt_bboxes.append(obj["bbox"])
                    gt_labels.append(obj["category_id"])

            if len(gt_bboxes) == 0:
                continue  # No valid objects in this image

            # Convert lists to tensors
            gt_bboxes = torch.stack(gt_bboxes).to(
                pred_boxes.device)  # Shape: (num_objects, 4)
            gt_labels = torch.tensor(gt_labels, dtype=torch.long).to(
                pred_scores.device)  # Shape: (num_objects,)

            # Ensure predictions match the number of ground truth objects
            pred_bboxes = pred_boxes[i, :gt_bboxes.shape[0], :]
            pred_cls_scores = pred_scores[i, :gt_labels.shape[0], :]

            # Compute losses
            bbox_loss = self.bbox_loss(pred_bboxes, gt_bboxes)
            cls_loss = self.cls_loss(pred_cls_scores, gt_labels)

            bbox_loss_total += bbox_loss
            cls_loss_total += cls_loss

        # Normalize loss by batch size
        total_loss = (self.bbox_weight * bbox_loss_total +
                      self.cls_weight * cls_loss_total) / batch_size
        return total_loss


def evaluate_model(model, data_loader, loss, metric):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    val_predictions, val_targets = [], []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = loss(outputs, targets)

            val_loss += loss.item()

            # Collect predictions & ground truth for mAP calculation
            val_predictions.append(
                {"boxes": outputs[:, :4], "scores": outputs[:, 4],
                 "labels": outputs[:, 5].int()})
            val_targets.append(
                {"boxes": targets[:, :4], "labels": targets[:, 4].int()})

    val_loss /= len(data_loader)
    # Compute mAP on validation set
    metric.update(val_predictions, val_targets)
    val_map = metric.compute()["map_50"]

    return val_loss, val_map


def train_model(model, train_loader, valid_loader, optimizer,
                epochs, early_stopping=None, wandb_log=None):
    loss_function = YOLONASLoss()
    train_losses, valid_losses, val_map_vec = [], [], []

    # Initialize mAP metric 50 and 75
    metric = MeanAveragePrecision(iou_thresholds=[0.5, 0.75])

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_predictions, train_targets = [], []

        progress_bar = tqdm(enumerate(train_loader),
                            desc=f"Epoch {epoch+1}/{epochs}",
                            total=len(train_loader))

        for batch_idx, (inputs, targets) in progress_bar:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = loss_function(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({"Train Loss": loss.item()})

            # Collect predictions & ground truth for mAP calculation
            train_predictions.append(
                {"boxes": outputs[:, :4], "scores": outputs[:, 4],
                 "labels": outputs[:, 5].int()})
            train_targets.append(
                {"boxes": targets[:, :4], "labels": targets[:, 4].int()})

        # Compute mAP on training set
        metric.update(train_predictions, train_targets)
        train_map = metric.compute()["map_50"]

        # Record training loss
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation step
        val_loss, val_map = evaluate_model(
            model, valid_loader, loss_function)
        valid_losses.append(val_loss)
        val_map_vec.append(val_map)

        # Print epoch summary
        print(f"Epoch {epoch+1}: Train Loss={train_loss: .4f}, \
                Train mAP={train_map: 0.4f}, \
                Val Loss={val_loss: .4f}, Val mAP={val_map: .2f} %")

        # Wandb log
        if wandb_log:
            wandb.log({"epoch": epoch+1,
                       "val_mAP": val_map, "train_mAP": train_map,
                       "train_loss": train_loss, "val_loss": val_loss})

        # Early stopping
        if early_stopping and early_stopping(val_loss, model):
            print("[INFO] Early stopping triggered.")
            break

    return train_losses, valid_losses, val_map_vec
