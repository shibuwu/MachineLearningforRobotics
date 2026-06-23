# Machine Learning for Robotics

Coursework from RBE 577 (Machine Learning for Robotics) at WPI. Four projects covering supervised learning for robot kinematics, physics-informed neural networks for contact prediction, deep reinforcement learning, and imitation learning for robotic manipulation.

Built with PyTorch, robosuite/robomimic, OpenAI Gym, and PyBullet.

---

## Projects

### Project 1 — Forward Kinematics Regression (`MLR_CS1/`)

Predicting end-effector pose (x, y, z, rx, ry, rz) of a UR10 robot arm from its 6 joint angles.

- Analytical (closed-form) and SGD-based linear regression on raw and engineered features (trig transforms expanding 6 inputs to 42)
- MLP regression (6 → 128 → 64 → 6, ReLU, Adam) with hyperparameter search over learning rates
- Trained on up to 80k samples, evaluated on a 100-sample linear trajectory
- Compared all five model variants on position and rotation error

**Stack:** NumPy, PyTorch, scikit-learn, pytransform3d

### Project 2 — Push Prediction with Physics-Informed Networks (`MLR_CS2/`)

Learning to predict object displacement (dx, dy, dθ) from planar push actions using three approaches:

- **Analytical physics model** — quasi-static push mechanics as a baseline
- **Pure neural network** — MLP with feature normalization, warmup + cosine LR schedule, L-BFGS fine-tuning
- **Hybrid model** — neural network augmented with physics model predictions as additional input features

Compared per-dimension MSE across all three, with trajectory rollout visualizations over sequential pushes.

**Stack:** PyTorch, NumPy, matplotlib

### Project 3 — Actor-Critic Deep RL (`MLR_CS3/`)

Two actor-critic implementations trained on continuous and visual control tasks:

- **A2C on LunarLander-v3** — separate actor/critic networks (256 hidden units), trained for 5k episodes with advantage estimation and gradient clipping
- **A3C on KukaDiverseObjectEnv** — asynchronous multi-worker training on a Kuka robot grasping task with image observations (128x128), using PyBullet

Also includes a from-scratch policy iteration solution on a tabular 3-state MDP to ground the theory before scaling up.

**Stack:** PyTorch, Gymnasium, PyBullet, pybullet_envs

### Project 4 — Imitation Learning for Robotic Manipulation (`MLR_CS4/`, `robosuite/`)

Training visuomotor policies to solve PickPlaceCan (Panda robot arm, robosuite) from human demonstrations:

- **Behavioral Cloning (BC)** — transformer-based policy trained on 222 demonstrations (gamepad + keyboard). Tuned architecture, batch size, LR schedule, and input modality. Best config (vision-only, no low-dim state) reached **100% success rate**
- **Diffusion Policy** — DDIM-based action generation (10 denoising steps) with EMA, trained on 50 curated demos. Faster inference than DDPM baseline (10x fewer steps)
- **Demo scaling study** — tested BC performance across 5, 10, 20, 50, and 200 demonstrations

All training ran on WPI's Turing cluster (A30/A100 GPUs). The `robosuite/` directory contains the simulation environment fork and robomimic training configs.

**Stack:** PyTorch, robosuite, robomimic, MuJoCo, TensorBoard

---

## Repository Structure

```
MLR_CS1/          Project 1 — forward kinematics regression
MLR_CS2/          Project 2 — push prediction (physics + neural + hybrid)
MLR_CS3/          Project 3 — A2C and A3C reinforcement learning
MLR_CS4/          Project 4 — behavioral cloning and diffusion policy
  ├── BC/           Behavioral cloning experiments
  └── Diffusion Policy/  Diffusion policy experiments
robosuite/        Simulation environment and training configs for Project 4
```

## Requirements

Each project has its own dependencies. Python 3.8+ and a CUDA-capable GPU are recommended across the board. See individual project directories for setup instructions.
