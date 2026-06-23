"""
Section 3.4: Compare all 5 models on ur10_linear_dataset
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from visualizer import plot_linear_trajectory
from lin_regression_analytic import AnalyticalLinearRegression
from lin_regression_sgd import SGDLinearRegression
from feature_engineering import engineer_features
from nn_regression import MLP


def main():
    # Load and prepare training data
    X, y = load_dataset("ur10_dataset.csv")
    X, y = X.values, y.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_feat = engineer_features(X_train)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Train all 5 models
    print("Training Analytic (raw)...")
    analytic_model = AnalyticalLinearRegression()
    analytic_model.fit(X_train, y_train)

    print("Training SGD (raw)...")
    sgd_model = SGDLinearRegression(learning_rate=0.001)
    sgd_model.fit(X_train, y_train, batch_size=32, epochs=200)

    print("Training Analytic (engineered)...")
    analytic_feat_model = AnalyticalLinearRegression()
    analytic_feat_model.fit(X_train_feat, y_train)

    print("Training SGD (engineered)...")
    sgd_feat_model = SGDLinearRegression(learning_rate=0.001)
    sgd_feat_model.fit(X_train_feat, y_train, batch_size=32, epochs=200)

    print("Training MLP...")
    mlp_model = MLP(input_size=6, hidden_sizes=[128, 64], output_size=6).to(device)
    mlp_model.fit(X_train, y_train, lr=0.001, batch_size=32, epochs=100, device=device)

    # Evaluate on ur10_linear_dataset
    print("\n=== Evaluating on ur10_linear_dataset ===\n")

    models = [
        ("1_analytic_raw", analytic_model, False),
        ("2_sgd_raw", sgd_model, False),
        ("3_analytic_engineered", analytic_feat_model, True),
        ("4_sgd_engineered", sgd_feat_model, True),
        ("5_mlp", mlp_model, False),
    ]

    for name, model, use_feat in models:
        print(f"\n{name}:")
        plot_linear_trajectory(model, use_engineered_features=use_feat)
        fig = plt.gcf()
        fig.savefig(f"plot_comparison_{name}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    main()
