from datasets import load_dataset
from helpers.metrics import compute_mse, compute_position_error, compute_rotation_error
from feature_engineering import engineer_features
import matplotlib.pyplot as plt

def plot_linear_trajectory(model, use_engineered_features=False):
    X_test_linear, y_test_linear = load_dataset("data/ur10_linear_dataset.csv")
    X_test_linear = X_test_linear.values
    y_test_linear = y_test_linear.values

    if use_engineered_features:
        X_test_linear = engineer_features(X_test_linear)

    y_pred_test = model.predict(X_test_linear)

    test_pos_error = compute_position_error(y_pred_test, y_test_linear)
    test_rot_error = compute_rotation_error(y_pred_test, y_test_linear)
    print(f"Position Error: {test_pos_error:.4f}")
    print(f"Rotation Error: {test_rot_error:.4f}")

    # Create subplots side by side
    fig = plt.figure(figsize=(14, 5))
    
    # Plot 1: Position (X, Y, Z)
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(y_test_linear[:,0], y_test_linear[:, 1], y_test_linear[:, 2], marker='s', linestyle='-', label='Ground Truth', markersize=10)
    ax1.plot(y_pred_test[:,0], y_pred_test[:, 1], y_pred_test[:, 2], marker='*', linestyle='-', label='Predicted') 
    ax1.legend()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Plot of Position')
    
    # Plot 2: Rotation (RX, RY, RZ)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(y_pred_test[:,3], y_pred_test[:, 4], y_pred_test[:, 5], marker='s', linestyle='-', label='Ground Truth', markersize=10)
    ax2.plot(y_test_linear[:,3], y_test_linear[:, 4], y_test_linear[:, 5], marker='*', linestyle='-', label='Predicted') 
    ax2.legend()
    ax2.set_xlabel('RX')
    ax2.set_ylabel('RY')
    ax2.set_zlabel('RZ')
    ax2.set_title('3D Plot of Rotation')


    plt.tight_layout()
    plt.show()

    #Take the best model and show me the plots.    
