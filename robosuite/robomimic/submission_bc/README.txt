Project 4: Imitation Learning Expert — BC Submission
======================================================
Student: Seeni Ramasamy
Course: RBE 577 - Machine Learning for Robotics
Partner: Shibani Senthilbabu (handling Diffusion Policy)

ENVIRONMENT SETUP
-----------------
conda activate robomimic_env
cd D:\MLR_CS4\robosuite\robomimic

TASK
----
PickPlaceCan with Panda robot
Dataset: best 50 demonstrations (demo_best50.hdf5)
Train/valid split: 45 train / 5 valid

BC RESULTS
----------
- BC baseline (2000 epochs): 0% success, severe overfitting
- BC improved (600 epochs): 0% success, stable validation loss 0.0365
- BC 1000 epochs: 0% success, stable validation loss 0.0384
- Best checkpoint: model_epoch_287_best_validation_0.0365.pth

DEMO SCALING RESULTS
--------------------
5 demos:  0% success
10 demos: 0% success
20 demos: 0% success
50 demos: 0% success (BC) / 10-20% (Diffusion - teammate)

EVALUATE BC MODEL
-----------------
python -m robomimic.scripts.run_trained_agent \
  --agent checkpoints/bc_model.pth \
  --n_rollouts 20 --horizon 400 --seed 0

VIEW TENSORBOARD
----------------
tensorboard --logdir logs/tensorboard/bc --bind_all
