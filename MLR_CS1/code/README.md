# Project 1: Joint Effort — Predicting the Pose

**Course:** Machine Learning for Robotics (RBE 577)
**Student:** Shibani, Senthilbabu

Predicts the end-effector pose (x, y, z, rx, ry, rz) of a UR10 robot from its 6 joint angles using linear regression and neural network models.

## Installation

Requires Python 3.8+ and PyTorch 2.4.1. GPU (CUDA) recommended for MLP training but CPU works as fallback.

```bash
conda create -n ml_env python=3.8.10 -y
conda activate ml_env
pip install -r requirements.txt
```

## Running the Code

All commands should be run from this directory (`code/`).

### Section 3.1 & 3.2: Linear Regression (Analytical + SGD, Raw + Engineered Features)

```bash
python3 run_experiments.py
```

Trains analytical and SGD linear regression models with both raw joint angles and engineered features across dataset sizes N = {10, 20, 50, ..., 80000}. Evaluates on a fixed 20% test set.

**Outputs:**
- `plot_loss_raw.png` — Train/test MSE vs dataset size (raw features)
- `plot_loss_engineered.png` — Train/test MSE vs dataset size (engineered features)
- `plot_raw_vs_engineered.png` — Test MSE comparison: raw vs engineered
- `plot_position_error.png` — Position error vs dataset size
- `plot_rotation_error.png` — Rotation error vs dataset size
- `plot_combined_error.png` — Combined error vs dataset size

### Section 3.3: MLP Hyperparameter Tuning

```bash
python3 run_mlp_experiments.py
```

Trains an MLP (6→128→64→6, ReLU, Adam) at learning rates {0.0001, 0.001, 0.01} with 70/10/20 train/val/test split.

**Outputs:**
- `plot_mlp_learning_rate.png` — Train/val loss vs epoch for each learning rate
- `plot_mlp_metrics.png` — MSE, position error, rotation error vs epoch

### Section 3.4: Overall Model Comparison

```bash
python3 run_comparison.py
```

Evaluates all 5 models (analytical raw, SGD raw, analytical engineered, SGD engineered, MLP) on the 100-sample linear trajectory dataset (`ur10_linear_dataset.csv`).

**Outputs:**
- `plot_comparison_1_analytic_raw.png`
- `plot_comparison_2_sgd_raw.png`
- `plot_comparison_3_analytic_engineered.png`
- `plot_comparison_4_sgd_engineered.png`
- `plot_comparison_5_mlp.png`

## File Structure

```
code/
├── helpers/
│   ├── loss.py              # MSE and custom weighted loss
│   └── metrics.py           # Position, rotation, combined error metrics
├── datasets.py              # Dataset loading and train/test splitting
├── visualizer.py            # 3D trajectory visualization
├── lin_regression_analytic.py   # Closed-form linear regression (numpy only)
├── lin_regression_sgd.py        # Mini-batch SGD linear regression (numpy only)
├── feature_engineering.py       # Trig feature transforms (6 → 42 features)
├── nn_regression.py             # PyTorch MLP regression model
├── run_experiments.py           # Sections 3.1 & 3.2 experiments
├── run_mlp_experiments.py       # Section 3.3 MLP experiments
├── run_comparison.py            # Section 3.4 model comparison
├── ur10_dataset.csv             # 100k samples (training data)
├── ur10_linear_dataset.csv      # 100 samples (linear trajectory test set)
├── requirements.txt             # Python dependencies
└── README.md
```

## Dependencies

- numpy, pandas, matplotlib — data handling and plotting
- torch 2.4.1 — MLP model (Section 3.3)
- scikit-learn — data utilities
- pytransform3d — rotation representations
- tqdm — progress bars
