import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from helpers.metrics import compute_mse, compute_position_error, compute_rotation_error
from lin_regression_analytic import AnalyticalLinearRegression
from lin_regression_sgd import SGDLinearRegression
from feature_engineering import engineer_features

# Dataset sizes to test
N_VALUES = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 80000]


def run_linear_regression_experiments():
    """Run experiments for Section 3.1 and 3.2"""
    X, y = load_dataset("ur10_dataset.csv")
    X, y = X.values, y.values

    # 80/20 split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {
        'analytic': {'train_loss': [], 'test_loss': [], 'pos_err': [], 'rot_err': []},
        'sgd': {'train_loss': [], 'test_loss': [], 'pos_err': [], 'rot_err': []},
        'analytic_feat': {'train_loss': [], 'test_loss': [], 'pos_err': [], 'rot_err': []},
        'sgd_feat': {'train_loss': [], 'test_loss': [], 'pos_err': [], 'rot_err': []},
    }

    for N in N_VALUES:
        if N > len(X_train_full):
            break
        print(f"\nTraining with N={N}")

        X_train = X_train_full[:N]
        y_train = y_train_full[:N]
        X_train_feat = engineer_features(X_train)
        X_test_feat = engineer_features(X_test)

        # Analytical (raw)
        model = AnalyticalLinearRegression()
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        results['analytic']['train_loss'].append(compute_mse(train_pred, y_train))
        results['analytic']['test_loss'].append(compute_mse(test_pred, y_test))
        pos_e = compute_position_error(test_pred, y_test)
        rot_e = compute_rotation_error(test_pred, y_test)
        results['analytic']['pos_err'].append(pos_e)
        results['analytic']['rot_err'].append(rot_e)
        print(f"  Analytic (raw):        MSE={results['analytic']['test_loss'][-1]:.4f}, Pos={pos_e:.4f}, Rot={rot_e:.4f}, Combined={pos_e + rot_e:.4f}")

        # SGD (raw)
        model = SGDLinearRegression(learning_rate=0.001)
        model.fit(X_train, y_train, batch_size=32, epochs=200)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        results['sgd']['train_loss'].append(compute_mse(train_pred, y_train))
        results['sgd']['test_loss'].append(compute_mse(test_pred, y_test))
        pos_e = compute_position_error(test_pred, y_test)
        rot_e = compute_rotation_error(test_pred, y_test)
        results['sgd']['pos_err'].append(pos_e)
        results['sgd']['rot_err'].append(rot_e)
        print(f"  SGD (raw):             MSE={results['sgd']['test_loss'][-1]:.4f}, Pos={pos_e:.4f}, Rot={rot_e:.4f}, Combined={pos_e + rot_e:.4f}")

        # Analytical (engineered features)
        model = AnalyticalLinearRegression()
        model.fit(X_train_feat, y_train)
        train_pred = model.predict(X_train_feat)
        test_pred = model.predict(X_test_feat)
        results['analytic_feat']['train_loss'].append(compute_mse(train_pred, y_train))
        results['analytic_feat']['test_loss'].append(compute_mse(test_pred, y_test))
        pos_e = compute_position_error(test_pred, y_test)
        rot_e = compute_rotation_error(test_pred, y_test)
        results['analytic_feat']['pos_err'].append(pos_e)
        results['analytic_feat']['rot_err'].append(rot_e)
        print(f"  Analytic (engineered): MSE={results['analytic_feat']['test_loss'][-1]:.4f}, Pos={pos_e:.4f}, Rot={rot_e:.4f}, Combined={pos_e + rot_e:.4f}")

        # SGD (engineered features)
        model = SGDLinearRegression(learning_rate=0.001)
        model.fit(X_train_feat, y_train, batch_size=32, epochs=200)
        train_pred = model.predict(X_train_feat)
        test_pred = model.predict(X_test_feat)
        results['sgd_feat']['train_loss'].append(compute_mse(train_pred, y_train))
        results['sgd_feat']['test_loss'].append(compute_mse(test_pred, y_test))
        pos_e = compute_position_error(test_pred, y_test)
        rot_e = compute_rotation_error(test_pred, y_test)
        results['sgd_feat']['pos_err'].append(pos_e)
        results['sgd_feat']['rot_err'].append(rot_e)
        print(f"  SGD (engineered):      MSE={results['sgd_feat']['test_loss'][-1]:.4f}, Pos={pos_e:.4f}, Rot={rot_e:.4f}, Combined={pos_e + rot_e:.4f}")

    return results


def plot_linear_results(results):
    """Plot results for linear regression experiments"""
    n_vals = N_VALUES[:len(results['analytic']['test_loss'])]

    # Plot 1: Train/Test loss vs N (raw features)
    plt.figure(figsize=(10, 6))
    plt.plot(n_vals, results['analytic']['train_loss'], 'b-o', label='Analytic Train')
    plt.plot(n_vals, results['analytic']['test_loss'], 'b--s', label='Analytic Test')
    plt.plot(n_vals, results['sgd']['train_loss'], 'r-o', label='SGD Train')
    plt.plot(n_vals, results['sgd']['test_loss'], 'r--s', label='SGD Test')
    plt.xscale('log')
    plt.xlabel('Training Dataset Size (N)')
    plt.ylabel('MSE Loss')
    plt.title('Linear Regression: Loss vs Dataset Size (Raw Features)')
    plt.legend()
    plt.grid(True)
    plt.savefig('plot_loss_raw.png')
    plt.show()

    # Plot 2: Train/Test loss vs N (engineered features)
    plt.figure(figsize=(10, 6))
    plt.plot(n_vals, results['analytic_feat']['train_loss'], 'b-o', label='Analytic Train')
    plt.plot(n_vals, results['analytic_feat']['test_loss'], 'b--s', label='Analytic Test')
    plt.plot(n_vals, results['sgd_feat']['train_loss'], 'r-o', label='SGD Train')
    plt.plot(n_vals, results['sgd_feat']['test_loss'], 'r--s', label='SGD Test')
    plt.xscale('log')
    plt.xlabel('Training Dataset Size (N)')
    plt.ylabel('MSE Loss')
    plt.title('Linear Regression: Loss vs Dataset Size (Engineered Features)')
    plt.legend()
    plt.grid(True)
    plt.savefig('plot_loss_engineered.png')
    plt.show()

    # Plot 3: Raw vs Engineered comparison
    plt.figure(figsize=(10, 6))
    plt.plot(n_vals, results['analytic']['test_loss'], 'b--', label='Analytic Raw')
    plt.plot(n_vals, results['analytic_feat']['test_loss'], 'b-', label='Analytic Engineered')
    plt.plot(n_vals, results['sgd']['test_loss'], 'r--', label='SGD Raw')
    plt.plot(n_vals, results['sgd_feat']['test_loss'], 'r-', label='SGD Engineered')
    plt.xscale('log')
    plt.xlabel('Training Dataset Size (N)')
    plt.ylabel('Test MSE Loss')
    plt.title('Raw vs Engineered Features Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('plot_raw_vs_engineered.png')
    plt.show()

    # Plot 4: Position Error vs N
    plt.figure(figsize=(10, 6))
    plt.plot(n_vals, results['analytic']['pos_err'], 'b-o', label='Analytic Raw')
    plt.plot(n_vals, results['sgd']['pos_err'], 'r-o', label='SGD Raw')
    plt.plot(n_vals, results['analytic_feat']['pos_err'], 'b--s', label='Analytic Engineered')
    plt.plot(n_vals, results['sgd_feat']['pos_err'], 'r--s', label='SGD Engineered')
    plt.xscale('log')
    plt.xlabel('Training Dataset Size (N)')
    plt.ylabel('Position Error')
    plt.title('Position Error vs Dataset Size')
    plt.legend()
    plt.grid(True)
    plt.savefig('plot_position_error.png')
    plt.show()

    # Plot 5: Rotation Error vs N
    plt.figure(figsize=(10, 6))
    plt.plot(n_vals, results['analytic']['rot_err'], 'b-o', label='Analytic Raw')
    plt.plot(n_vals, results['sgd']['rot_err'], 'r-o', label='SGD Raw')
    plt.plot(n_vals, results['analytic_feat']['rot_err'], 'b--s', label='Analytic Engineered')
    plt.plot(n_vals, results['sgd_feat']['rot_err'], 'r--s', label='SGD Engineered')
    plt.xscale('log')
    plt.xlabel('Training Dataset Size (N)')
    plt.ylabel('Rotation Error')
    plt.title('Rotation Error vs Dataset Size')
    plt.legend()
    plt.grid(True)
    plt.savefig('plot_rotation_error.png')
    plt.show()

    # Plot 6: Combined Error vs N
    plt.figure(figsize=(10, 6))
    analytic_combined = [p + r for p, r in zip(results['analytic']['pos_err'], results['analytic']['rot_err'])]
    sgd_combined = [p + r for p, r in zip(results['sgd']['pos_err'], results['sgd']['rot_err'])]
    analytic_feat_combined = [p + r for p, r in zip(results['analytic_feat']['pos_err'], results['analytic_feat']['rot_err'])]
    sgd_feat_combined = [p + r for p, r in zip(results['sgd_feat']['pos_err'], results['sgd_feat']['rot_err'])]
    plt.plot(n_vals, analytic_combined, 'b-o', label='Analytic Raw')
    plt.plot(n_vals, sgd_combined, 'r-o', label='SGD Raw')
    plt.plot(n_vals, analytic_feat_combined, 'b--s', label='Analytic Engineered')
    plt.plot(n_vals, sgd_feat_combined, 'r--s', label='SGD Engineered')
    plt.xscale('log')
    plt.xlabel('Training Dataset Size (N)')
    plt.ylabel('Combined Error (Position + Rotation)')
    plt.title('Combined Error vs Dataset Size')
    plt.legend()
    plt.grid(True)
    plt.savefig('plot_combined_error.png')
    plt.show()


if __name__ == "__main__":
    print("Running linear regression experiments...")
    results = run_linear_regression_experiments()
    plot_linear_results(results)
    print("Done! Plots saved.")
