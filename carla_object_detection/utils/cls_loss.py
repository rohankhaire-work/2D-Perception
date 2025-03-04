import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
        """
        Focal Loss implementation.

        Args:
            gamma (float): Focusing parameter to reduce easy sample weight.
            alpha (float): Balancing factor for positive and negative classes.
            reduction (str): "mean", "sum", or "none".
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(
            reduction="none")  # No reduction yet

    def forward(self, input, target):
        """
        Compute focal loss.

        Args:
            input (Tensor): Predicted logits (N, C).
            target (Tensor): class label

        Returns:
            Tensor: Computed focal loss.
        """
        device = input.device

        target = target.to(device)

        # Create a zero tensor of shape (N, num_classes)
        multi_hot = torch.zeros(6, dtype=torch.float32).to(device)

        # Iterate through batch and set appropriate indices to 1
        for class_label in (target):
            multi_hot[class_label] = 1

        multihot_final = multi_hot.expand(len(input), -1)
        # Compute BCE loss (without reduction)
        bce_loss = self.bce_loss(input, multihot_final)

        # Compute pt (probability of the true class)
        probas = torch.sigmoid(input)
        # pt = p if y=1 else (1-p)
        pt = probas * multihot_final + (1 - probas) * (1 - multihot_final)

        # Compute Focal Loss scaling factor
        focal_weight = (1 - pt) ** self.gamma

        # Compute Alpha balancing factor
        alpha_factor = self.alpha * multihot_final + \
            (1 - self.alpha) * (1 - multihot_final)

        # Apply Focal Loss weighting
        focal_loss = alpha_factor * focal_weight * bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss  # No reduction
