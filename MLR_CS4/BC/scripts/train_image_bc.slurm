#!/bin/bash
#SBATCH --job-name=p4_image_bc
#SBATCH --account=rbe577
#SBATCH --partition=academic
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=/home/ssenthilbabu/rbe577_p4/logs/image_bc_%j.log

source /etc/profile.d/modules.sh
module load miniconda3/25.1.1/24g7bpu
source activate robomimic_v15
export MUJOCO_GL=egl
cd ~/rbe577_p4

echo "=== JOB $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi --query-gpu=name --format=csv,noheader
pip install tensorboardX 2>&1 | tail -1

# Step 1: Fix the train/valid split (currently 23/199, should be ~200/22)
python << 'PYEOF'
import h5py, numpy as np

ds_path = 'datasets/all_pickplace_image.hdf5'
f = h5py.File(ds_path, 'a')
demos = sorted(list(f['data'].keys()))
n = len(demos)
print(f"Total demos: {n}")

# Check current split
old_train = len(f['mask']['train'][()])
old_valid = len(f['mask']['valid'][()])
print(f"Old split: train={old_train}, valid={old_valid}")

# 90/10 split with seed 42
np.random.seed(42)
perm = np.random.permutation(n)
n_train = int(0.9 * n)
train_demos = [demos[i] for i in perm[:n_train]]
valid_demos = [demos[i] for i in perm[n_train:]]

# Overwrite the mask
del f['mask']['train']
del f['mask']['valid']
f['mask'].create_dataset('train', data=np.array(train_demos, dtype='S'))
f['mask'].create_dataset('valid', data=np.array(valid_demos, dtype='S'))

print(f"New split: train={len(train_demos)}, valid={len(valid_demos)}")
f.close()
PYEOF

# Step 2: Generate image BC config
python << 'PYEOF'
import json, copy

with open('configs/bc_merged.json') as f:
    base = json.load(f)

c = copy.deepcopy(base)
c['experiment']['name'] = 'bc_image_transformer'
c['train']['output_dir'] = '/home/ssenthilbabu/rbe577_p4/results/bc_image'

# Dataset already has image obs — use it directly
c['train']['hdf5_cache_mode'] = 'low_dim'
c['train']['num_data_workers'] = 2
c['train']['batch_size'] = 16
c['train']['num_epochs'] = 6000
c['train']['seq_length'] = 10
c['train']['frame_stack'] = 10

# Observations: low-dim + images
c['observation']['modalities']['obs']['low_dim'] = [
    'robot0_eef_pos',
    'robot0_eef_quat',
    'robot0_gripper_qpos',
    'object'
]
c['observation']['modalities']['obs']['rgb'] = [
    'agentview_image',
    'robot0_eye_in_hand_image'
]

# Image encoder: ResNet18 + SpatialSoftmax + CropRandomizer
c['observation']['encoder']['rgb'] = {
    'core_class': 'VisualCore',
    'core_kwargs': {
        'feature_dimension': 64,
        'backbone_class': 'ResNet18Conv',
        'backbone_kwargs': {
            'pretrained': False,
            'input_coord_conv': False
        },
        'pool_class': 'SpatialSoftmax',
        'pool_kwargs': {
            'num_kp': 32,
            'learnable_temperature': False,
            'temperature': 1.0,
            'noise_std': 0.0
        }
    },
    'obs_randomizer_class': 'CropRandomizer',
    'obs_randomizer_kwargs': {
        'crop_height': 76,
        'crop_width': 76,
        'num_crops': 1,
        'pos_enc': False
    }
}

# Transformer — same as Tune 5
c['algo']['transformer']['enabled'] = True
c['algo']['transformer']['context_length'] = 10
c['algo']['transformer']['embed_dim'] = 256
c['algo']['transformer']['num_layers'] = 4
c['algo']['transformer']['num_heads'] = 8
c['algo']['transformer']['emb_dropout'] = 0.2
c['algo']['transformer']['attn_dropout'] = 0.2
c['algo']['transformer']['block_output_dropout'] = 0.2
c['algo']['transformer']['sinusoidal_embedding'] = False
c['algo']['transformer']['activation'] = 'gelu'
c['algo']['transformer']['supervise_all_steps'] = True
c['algo']['transformer']['nn_parameter_for_timesteps'] = True

c['algo']['rnn']['enabled'] = False
c['algo']['gmm']['enabled'] = False

c['algo']['loss'] = {'l2_weight': 0.3, 'l1_weight': 0.7, 'cos_weight': 0.0}
c['algo']['actor_layer_dims'] = [1024, 1024]

# Optimizer: cosine LR with warmup
c['algo']['optim_params']['policy']['optimizer_type'] = 'adamw'
c['algo']['optim_params']['policy']['learning_rate'] = {
    'initial': 0.0001,
    'scheduler_type': 'cosine',
    'epoch_schedule': [],
    'decay_factor': 0.1,
    'step_every_batch': True,
    'warmup_steps': 500
}
c['algo']['optim_params']['policy']['regularization'] = {'L2': 0.001}

# Rollout & saving
c['experiment']['save']['every_n_epochs'] = 500
c['experiment']['save']['on_best_rollout_success_rate'] = True
c['experiment']['rollout']['enabled'] = True
c['experiment']['rollout']['n'] = 10
c['experiment']['rollout']['horizon'] = 1000
c['experiment']['rollout']['rate'] = 500
c['experiment']['rollout']['terminate_on_success'] = True
c['experiment']['render_video'] = True
c['experiment']['logging']['log_tb'] = True
c['train']['max_grad_norm'] = 1.0

with open('configs/bc_image.json', 'w') as f:
    json.dump(c, f, indent=4)
print('Image BC config written')
PYEOF

mkdir -p ~/rbe577_p4/results/bc_image

echo "=== Training Image-Based BC (6000 epochs) ==="
yes | python robomimic/robomimic/scripts/train.py \
    --config configs/bc_image.json \
    --dataset datasets/all_pickplace_image.hdf5 \
    --name bc_image_transformer || echo "Image BC FAILED"

echo "=== DONE ==="
