import torch
import torch.nn as nn


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, predictions, targets):
        return self.criterion(predictions, targets)


class CustomLoss(nn.Module):
    def __init__(self, position_weight=1.0, rotation_weight=1.0):
        super(CustomLoss, self).__init__()
        self.position_weight = position_weight
        self.rotation_weight = rotation_weight

    def forward(self, predictions, targets):
        # Split predictions and targets into position and rotation components
        pos_pred, rot_pred = predictions[:, :3], predictions[:, 3:]
        pos_true, rot_true = targets[:, :3], targets[:, 3:]

        # Calculate MSE for position and rotation separately
        pos_loss = torch.mean((pos_pred - pos_true) ** 2)
        rot_loss = torch.mean((rot_pred - rot_true) ** 2)

        # Combine losses with weights
        total_loss = self.position_weight * pos_loss + self.rotation_weight * rot_loss
        return total_loss
