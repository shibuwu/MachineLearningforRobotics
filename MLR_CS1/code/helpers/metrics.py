import numpy as np
import torch


def compute_mse(predictions, targets):
    return np.mean((predictions - targets) ** 2)

def compute_position_error(predictions, targets):
    pos_pred = predictions[:, :3]
    pos_true = targets[:, :3]
    return np.mean(np.sqrt(np.sum((pos_pred - pos_true) ** 2, axis=1)))

def compute_rotation_error(predictions, targets):
    rot_pred = predictions[:, 3:]
    rot_true = targets[:, 3:]
    return np.mean(np.sqrt(np.sum((rot_pred - rot_true) ** 2, axis=1)))
