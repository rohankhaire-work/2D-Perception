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
        # Summed loss for efficiency
        self.bbox_loss = nn.MSELoss(reduction='sum')
        self.cls_loss = nn.CrossEntropyLoss(reduction='sum')
        self.bbox_weight = bbox_weight
        self.cls_weight = cls_weight

    def forward(self, outputs, targets):
        """
        Compute YOLO-NAS loss for the entire batch.
        :param outputs: Tuple (pred_boxes, pred_scores)
        :param targets: List of lists containing ground truth data for each image.
        :return: Total batch loss (bbox loss + class loss)
        """
        pred_boxes, pred_scores = outputs
        print(outputs)
        if isinstance(pred_boxes, tuple):
            conf_scores = pred_boxes[1]
            pred_boxes = pred_boxes[0]
        if isinstance(pred_scores, tuple):
            label_scores = pred_scores[0]

        device = pred_boxes.device  # Ensure everything runs on GPU if available

        all_gt_bboxes = []
        all_gt_labels = []
        all_pred_bboxes = []
        all_pred_scores = []

        for i, image_targets in enumerate(targets):
            if not image_targets:
                continue  # Skip images with no objects

            # Extract ground truth boxes and labels
            gt_bboxes = torch.tensor(
                [obj["bbox"] for obj in image_targets], dtype=torch.float32, device=device
            )
            gt_labels = torch.tensor(
                [obj["category_id"] for obj in image_targets], dtype=torch.long, device=device
            )

            # Select top-k predictions for this image
            num_gt = gt_bboxes.shape[0]
            # Number of predictions per image
            num_pred = pred_boxes[i].shape[1]
            # Ensure we don't select more than available predictions
            k = min(num_gt, num_pred)

            # Select top-k predictions based on confidence scores
            if num_pred > k:
                topk_indices = torch.argsort(
                    conf_scores[i, :, 0], descending=True)[:k]
                selected_pred_bboxes = pred_boxes[i, topk_indices, :]
                selected_pred_scores = conf_scores[i, topk_indices, :]
            else:
                selected_pred_bboxes = pred_boxes[i, :num_gt, :]
                selected_pred_scores = conf_scores[i, :num_gt, :]

            # Append data to lists
            all_gt_bboxes.append(gt_bboxes)
            all_gt_labels.append(gt_labels)
            all_pred_bboxes.append(selected_pred_bboxes)
            all_pred_scores.append(selected_pred_scores)

        if not all_gt_bboxes:
            # No objects, return zero loss
            return torch.tensor(0.0, device=device)

        # Stack all data
        gt_bboxes = torch.cat(all_gt_bboxes, dim=0)
        gt_labels = torch.cat(all_gt_labels, dim=0)
        pred_bboxes = torch.cat(all_pred_bboxes, dim=0)
        pred_scores = torch.cat(all_pred_scores, dim=0)

        # Compute losses
        bbox_loss = self.bbox_loss(pred_bboxes, gt_bboxes)
        cls_loss = self.cls_loss(pred_scores, gt_labels)

        # Normalize by batch size (avoiding division by zero)
        total_loss = (self.bbox_weight * bbox_loss +
                      self.cls_weight * cls_loss) / max(1, len(targets))
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
            pred_boxes, pred_scores = outputs

            # Determine class probabilities and max scores
            class_probs = torch.softmax(pred_scores[0], dim=-1)
            max_scores, pred_labels = torch.max(class_probs, dim=-1)

            train_predictions.append({
                "boxes": pred_boxes[0],
                "scores": max_scores,
                "labels": pred_labels.int()})

            for t in targets[0]:
                train_targets.append(
                    {"boxes": t["bbox"], "labels": t["category_id"]})

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
