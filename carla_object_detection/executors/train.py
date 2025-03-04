from executors.callback import process_prediction, process_groundtruth
from executors.loss import compute_loss
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class YOLONASLoss:
    def __init__(self):
        pass


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


def evaluate_model(model, data_loader):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)

            # Prccess the Prediction and Target
            gt_boxes, gt_labels = process_groundtruth(targets)
            pred_boxes, pred_scores, logits = process_prediction(outputs)
            # Calculate loss
            # loss conssit of regression and classification
            loss = compute_loss(pred_boxes, pred_scores,
                                logits, gt_boxes, gt_labels)

            val_loss += loss

        val_loss /= len(data_loader)

    return val_loss


def train_model(model, train_loader, valid_loader, optimizer,
                epochs, early_stopping=None, wandb_log=None):

    train_losses, valid_losses = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(enumerate(train_loader),
                            desc=f"Epoch {epoch+1}/{epochs}",
                            total=len(train_loader))

        for batch_idx, (inputs, targets) in progress_bar:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)

            # Prccess the Prediction and Target
            pred_boxes, pred_scores, logits = process_prediction(outputs)
            gt_boxes, gt_labels = process_groundtruth(targets)

            # Calculate loss
            # loss conssit of regression and classification
            loss = compute_loss(pred_boxes, pred_scores,
                                logits, gt_boxes, gt_labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss
            progress_bar.set_postfix({"Train Loss": loss.item()})

        # Record training loss
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation step
        val_loss = evaluate_model(
            model, valid_loader)
        valid_losses.append(val_loss)

        # Print epoch summary
        print(f"Epoch {epoch+1}: Train Loss={train_loss: .4f}, \
                Val Loss={val_loss: .4f}")

        # Wandb log
        if wandb_log:
            wandb.log({"epoch": epoch+1,
                       "train_loss": train_loss, "val_loss": val_loss})

        # Early stopping
        if early_stopping and early_stopping(val_loss, model):
            print("[INFO] Early stopping triggered.")
            break

    return train_losses, valid_losses
