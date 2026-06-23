Project 4: Imitation Learning Expert
=====================================
Student: Seeni Ramasamy
Course: RBE 577 - Machine Learning for Robotics

ENVIRONMENT SETUP
-----------------
conda activate robomimic_env
cd D:\MLR_CS4\robosuite\robomimic

TASKS
-----
- Lift task (custom dataset): bc_model.pth
- PickPlaceCan (best50 demos): diffusion_model.pth

BEST RESULTS
------------
- BC Lift: 20% success rate (4/20 rollouts)
- Diffusion PickPlaceCan improved: 20-60% success rate

EVALUATE BEST DIFFUSION MODEL
-------------------------------
python -m robomimic.scripts.run_trained_agent \
  --agent checkpoints/diffusion_model.pth \
  --n_rollouts 20 --horizon 400 --seed 0

EVALUATE BC MODEL
-----------------
python -m robomimic.scripts.run_trained_agent \
  --agent checkpoints/bc_model.pth \
  --n_rollouts 20 --horizon 400 --seed 0

VIEW TENSORBOARD
----------------
tensorboard --logdir logs/tensorboard --bind_all

DATASET SPLITS
--------------
demo_best50.hdf5 contains mask keys: train, valid, demo5, demo10, demo20
