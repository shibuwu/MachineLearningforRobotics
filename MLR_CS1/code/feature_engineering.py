import numpy as np


def engineer_features(angles):
    """Engineer features for robot kinematics based on forward kinematics equations.

    Creates features from joint angles that better capture the nonlinear relationships in robot forward kinematics.

    Args:
        angles (np.ndarray): Input joint angles array of shape (n_samples, 6)

    Returns:
        np.ndarray: Engineered features array of shape (n_samples, 42)

    Example:
        >>> angles = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
        >>> features = engineer_features(angles)
        >>> print(features.shape)
        (1, 42)
    """
    features = []

    # sin/cos of each angle (12 features)
    for i in range(6):
        features.append(np.sin(angles[:, i:i+1]))
        features.append(np.cos(angles[:, i:i+1]))

    # sin/cos of cumulative angle sums (10 features)
    cumsum = np.cumsum(angles, axis=1)
    for i in range(1, 6):  # θ1+θ2, θ1+θ2+θ3, etc.
        features.append(np.sin(cumsum[:, i:i+1]))
        features.append(np.cos(cumsum[:, i:i+1]))

    # Product terms: sin(θi)*cos(θi) for each angle (6 features)
    for i in range(6):
        features.append((np.sin(angles[:, i]) * np.cos(angles[:, i])).reshape(-1, 1))

    # Cross terms: sin(θi)*cos(θj) for adjacent pairs (10 features)
    for i in range(5):
        features.append((np.sin(angles[:, i]) * np.cos(angles[:, i+1])).reshape(-1, 1))
        features.append((np.cos(angles[:, i]) * np.sin(angles[:, i+1])).reshape(-1, 1))

    # Additional cross terms to reach 42 (4 more features)
    features.append((np.sin(angles[:, 0]) * np.sin(angles[:, 2])).reshape(-1, 1))
    features.append((np.cos(angles[:, 0]) * np.cos(angles[:, 2])).reshape(-1, 1))
    features.append((np.sin(angles[:, 1]) * np.sin(angles[:, 3])).reshape(-1, 1))
    features.append((np.cos(angles[:, 1]) * np.cos(angles[:, 3])).reshape(-1, 1))

    return np.hstack(features)