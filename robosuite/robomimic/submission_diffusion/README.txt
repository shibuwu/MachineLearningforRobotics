Project 4: Imitation Learning Expert — Diffusion Policy Submission
===================================================================
Student: Seeni Ramasamy
Course: RBE 577 - Machine Learning for Robotics

BEST MODEL
----------
Algorithm: Diffusion Policy (improved)
Task: PickPlaceCan with Panda robot
Dataset: best 50 demonstrations
Best checkpoint: model_epoch_110_best_validation_0.0193.pth
Success Rate: 10-20% (best observed: 60% over 5 rollouts)

KEY IMPROVEMENTS OVER BASELINE
-------------------------------
- DDIM (10 steps) instead of DDPM (100 steps) — 10x faster inference
- EMA enabled (power=0.75)
- Observation normalization enabled
- Action normalization (min_max)
- Reduced epochs: 800 vs 2000
- Gradient clipping: max_grad_norm=1.0

EVALUATE MODEL
--------------
python -m robomimic.scripts.run_trained_agent \
  --agent checkpoints/diffusion_model.pth \
  --n_rollouts 20 --horizon 400 --seed 0

VIEW TENSORBOARD
----------------
tensorboard --logdir logs/tensorboard/diffusion_policy --bind_all
