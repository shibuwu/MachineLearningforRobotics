import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from helpers.metrics import compute_mse, compute_position_error, compute_rotation_error
from nn_regression import MLP, convert_to_tensor
from torch.utils.data import DataLoader, TensorDataset
from helpers.loss import CustomLoss


def train_mlp_with_validation(X_train, y_train, X_val, y_val, lr=0.001, batch_size=32, epochs=100, device="cuda"):
    """Train MLP and track train/val loss per epoch"""
    model = MLP(input_size=6, hidden_sizes=[128, 64], output_size=6).to(device)

    X_train_tensor = convert_to_tensor(X_train, device)
    y_train_tensor = convert_to_tensor(y_train, device)
    X_val_tensor = convert_to_tensor(X_val, device)
    y_val_tensor = convert_to_tensor(y_val, device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = CustomLoss(position_weight=1.0, rotation_weight=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor)
            val_loss = criterion(val_pred, y_val_tensor)
            val_losses.append(val_loss.item())

    return model, train_losses, val_losses


def run_hyperparameter_tuning():
    """Run MLP hyperparameter tuning experiments"""
    X, y = load_dataset("ur10_dataset.csv")
    X, y = X.values, y.values

    # 70/10/20 split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42)  # 0.125 of 80% = 10%

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Hyperparameter: Learning Rate
    learning_rates = [0.0001, 0.001, 0.01]
    lr_results = {}

    for lr in learning_rates:
        print(f"Training with lr={lr}")
        model, train_losses, val_losses = train_mlp_with_validation(
            X_train, y_train, X_val, y_val, lr=lr, batch_size=32, epochs=100, device=device
        )
        lr_results[lr] = {'train': train_losses, 'val': val_losses, 'model': model}

    # Plot learning rate comparison
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for lr in learning_rates:
        plt.plot(lr_results[lr]['train'], label=f'lr={lr}')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Epoch (Different Learning Rates)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for lr in learning_rates:
        plt.plot(lr_results[lr]['val'], label=f'lr={lr}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss vs Epoch (Different Learning Rates)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plot_mlp_learning_rate.png')
    plt.show()

    # Final metrics comparison
    print("\n=== Final Test Set Performance ===")
    best_model = None
    best_val_loss = float('inf')

    for lr in learning_rates:
        model = lr_results[lr]['model']
        model.eval()
        with torch.no_grad():
            test_pred = model.predict(X_test, device=device)
        pos_err = compute_position_error(test_pred, y_test)
        rot_err = compute_rotation_error(test_pred, y_test)
        mse = compute_mse(test_pred, y_test)
        final_val = lr_results[lr]['val'][-1]

        print(f"LR={lr}: MSE={mse:.4f}, Pos_Err={pos_err:.4f}, Rot_Err={rot_err:.4f}")

        if final_val < best_val_loss:
            best_val_loss = final_val
            best_model = model

    return best_model, lr_results


def plot_metrics_over_epochs(X_train, y_train, X_val, y_val, X_test, y_test, lr=0.001, epochs=100, device="cuda"):
    """Plot all metrics over training epochs"""
    model = MLP(input_size=6, hidden_sizes=[128, 64], output_size=6).to(device)

    X_train_tensor = convert_to_tensor(X_train, device)
    y_train_tensor = convert_to_tensor(y_train, device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    criterion = CustomLoss(position_weight=1.0, rotation_weight=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_mse, val_mse = [], []
    train_pos, val_pos = [], []
    train_rot, val_rot = [], []

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_pred = model.predict(X_train, device=device)
            val_pred = model.predict(X_val, device=device)

        train_mse.append(compute_mse(train_pred, y_train))
        val_mse.append(compute_mse(val_pred, y_val))
        train_pos.append(compute_position_error(train_pred, y_train))
        val_pos.append(compute_position_error(val_pred, y_val))
        train_rot.append(compute_rotation_error(train_pred, y_train))
        val_rot.append(compute_rotation_error(val_pred, y_val))

    # Plot metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(train_mse, label='Train')
    axes[0].plot(val_mse, label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('MSE vs Epoch')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(train_pos, label='Train')
    axes[1].plot(val_pos, label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Position Error')
    axes[1].set_title('Position Error vs Epoch')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(train_rot, label='Train')
    axes[2].plot(val_rot, label='Validation')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Rotation Error')
    axes[2].set_title('Rotation Error vs Epoch')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('plot_mlp_metrics.png')
    plt.show()

    # Final test performance
    test_pred = model.predict(X_test, device=device)
    print("\n=== Final Test Set Metrics ===")
    print(f"MSE: {compute_mse(test_pred, y_test):.4f}")
    print(f"Position Error: {compute_position_error(test_pred, y_test):.4f}")
    print(f"Rotation Error: {compute_rotation_error(test_pred, y_test):.4f}")

    return model


if __name__ == "__main__":
    X, y = load_dataset("ur10_dataset.csv")
    X, y = X.values, y.values

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=== Hyperparameter Tuning ===")
    best_model, lr_results = run_hyperparameter_tuning()

    print("\n=== Metrics Over Epochs ===")
    model = plot_metrics_over_epochs(X_train, y_train, X_val, y_val, X_test, y_test, lr=0.001, epochs=100, device=device)

    print("\nDone! Plots saved.")
