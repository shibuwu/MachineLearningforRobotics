RBE 577 Project 4 - Imitation Learning Expert
Shibani Senthilbabu & Seenivasa Ramasamy

Task: PickPlaceCan with Panda robot (robosuite v1.5.2 + robomimic)

Directory Structure:
  report/                 - LaTeX source and figures
  code/training_configs/  - JSON configs for all experiments (4 image BC + 4 demo scaling)
  checkpoints/            - Best model checkpoints (baseline 90%, best 100%)
  tensorboard/            - TensorBoard event files for all runs
    bc_image/             - Baseline (low-dim + images)
    bc_image_tuneA/       - Tune A (smaller transformer)
    bc_image_tuneB/       - Tune B (batch=32, multistep LR)
    bc_image_tuneC/       - Tune C (images only, best)
    bc_demo_5/            - Demo scaling: 5 demos
    bc_demo_10/           - Demo scaling: 10 demos
    bc_demo_20/           - Demo scaling: 20 demos
    bc_demo_50/           - Demo scaling: 50 demos
  videos/                 - Rollout videos for all 4 image BC models
  logs/                   - Training logs for all 4 image BC experiments
  scripts/                - Custom scripts (gamepad driver, demo merger, SLURM, figures)

Experiments:
  Baseline    - Large TF (256/4/8) + low-dim + images     -> 90% success (epoch 1000)
  Tune A      - Smaller TF (128/2/4), dropout=0.1         -> 90% success (epoch 1000)
  Tune B      - Batch=32, multistep LR (3e-4)             -> 90% success (epoch 500)
  Tune C      - Images only (no low-dim state)             -> 100% success (epoch 2000)

Demo Scaling (Tune C architecture, 300 epochs):
  5 demos     -> 10% peak success
  10 demos    -> 50% peak success
  20 demos    -> 30% peak success
  50 demos    -> 40% peak success
  200 demos   -> 100% peak success (full training)

Best Result: 100% success rate (Tune C, epoch 2000)
  Images only (no low-dim state), large transformer (256/4/8),
  cosine LR (1e-4) + 500 warmup, dropout=0.2, L2=0.001

Environment:
  Conda: robomimic_v15 (Python 3.10, PyTorch 2.1.0+cu118, mujoco 3.8.0)
  HPC: WPI Turing cluster, NVIDIA A30/A100 GPUs, 24GB RAM
  Data: 222 demos (175 gamepad + 47 keyboard), 90/10 train/val split
