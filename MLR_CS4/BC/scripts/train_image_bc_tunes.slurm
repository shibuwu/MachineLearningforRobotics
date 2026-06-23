#!/bin/bash
#SBATCH --job-name=p4_img_tune
#SBATCH --account=rbe577
#SBATCH --partition=academic
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=/home/ssenthilbabu/rbe577_p4/logs/img_tune_%j.log

source /etc/profile.d/modules.sh
module load miniconda3/25.1.1/24g7bpu
source activate robomimic_v15
export MUJOCO_GL=egl
cd ~/rbe577_p4

echo "=== JOB $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi --query-gpu=name --format=csv,noheader
pip install tensorboardX 2>&1 | tail -1

# Generate all tune configs
python << 'PYEOF'
import json, copy

with open('configs/bc_image.json') as f:
    base = json.load(f)

base['train']['num_epochs'] = 2000

# ---- Tune A: Smaller transformer (128/2/4), less regularization ----
# Tests whether the smaller model works better with images (less overfitting risk)
c = copy.deepcopy(base)
c['experiment']['name'] = 'bc_image_tuneA'
c['train']['output_dir'] = '/home/ssenthilbabu/rbe577_p4/results/bc_image_tuneA'
c['algo']['transformer']['embed_dim'] = 128
c['algo']['transformer']['num_layers'] = 2
c['algo']['transformer']['num_heads'] = 4
c['algo']['transformer']['emb_dropout'] = 0.1
c['algo']['transformer']['attn_dropout'] = 0.1
c['algo']['transformer']['block_output_dropout'] = 0.1
c['algo']['optim_params']['policy']['regularization'] = {'L2': 0.0001}
with open('configs/bc_image_tuneA.json', 'w') as f:
    json.dump(c, f, indent=4)
print('Tune A: smaller transformer (128/2/4), dropout=0.1, L2=0.0001')

# ---- Tune B: Larger batch (32) + multistep LR ----
# Tests batch size effect and different LR schedule with images
c = copy.deepcopy(base)
c['experiment']['name'] = 'bc_image_tuneB'
c['train']['output_dir'] = '/home/ssenthilbabu/rbe577_p4/results/bc_image_tuneB'
c['train']['batch_size'] = 32
c['algo']['optim_params']['policy']['learning_rate'] = {
    'initial': 0.0003,
    'scheduler_type': 'multistep',
    'epoch_schedule': [100, 500, 1000, 2000, 3000],
    'decay_factor': 0.2,
    'step_every_batch': False,
    'warmup_steps': 0
}
with open('configs/bc_image_tuneB.json', 'w') as f:
    json.dump(c, f, indent=4)
print('Tune B: batch=32, multistep LR (3e-4), same large transformer')

# ---- Tune C: Images only, no low-dim observations ----
# Tests whether vision alone is sufficient without state info
c = copy.deepcopy(base)
c['experiment']['name'] = 'bc_image_tuneC'
c['train']['output_dir'] = '/home/ssenthilbabu/rbe577_p4/results/bc_image_tuneC'
c['observation']['modalities']['obs']['low_dim'] = []
c['observation']['modalities']['obs']['rgb'] = [
    'agentview_image',
    'robot0_eye_in_hand_image'
]
with open('configs/bc_image_tuneC.json', 'w') as f:
    json.dump(c, f, indent=4)
print('Tune C: images only (no low-dim obs), same large transformer')

PYEOF

# Tune A already ran — skipping

# ---- Run Tune B ----
mkdir -p ~/rbe577_p4/results/bc_image_tuneB
echo "=== Tune B: Larger Batch + Multistep LR ==="
yes | python robomimic/robomimic/scripts/train.py \
    --config configs/bc_image_tuneB.json \
    --dataset datasets/all_pickplace_image.hdf5 \
    --name bc_image_tuneB || echo "Tune B FAILED"

# ---- Run Tune C ----
mkdir -p ~/rbe577_p4/results/bc_image_tuneC
echo "=== Tune C: Images Only (no low-dim) ==="
yes | python robomimic/robomimic/scripts/train.py \
    --config configs/bc_image_tuneC.json \
    --dataset datasets/all_pickplace_image.hdf5 \
    --name bc_image_tuneC || echo "Tune C FAILED"

echo "=== ALL TUNES DONE ==="
