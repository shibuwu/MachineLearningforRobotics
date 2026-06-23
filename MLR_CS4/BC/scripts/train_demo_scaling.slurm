#!/bin/bash
#SBATCH --job-name=p4_demo_scale
#SBATCH --account=rbe577
#SBATCH --partition=academic
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=6:00:00
#SBATCH --output=/home/ssenthilbabu/rbe577_p4/logs/demo_scale_%j.log

source /etc/profile.d/modules.sh
module load miniconda3/25.1.1/24g7bpu
source activate robomimic_v15
export MUJOCO_GL=egl
cd ~/rbe577_p4

echo "=== JOB $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi --query-gpu=name --format=csv,noheader
pip install tensorboardX 2>&1 | tail -1

# Create subset masks and configs (idempotent, safe to run from any job)
python << 'PYEOF'
import h5py, numpy as np, json, copy

ds_path = 'datasets/all_pickplace_image.hdf5'
f = h5py.File(ds_path, 'a')
demos = sorted(list(f['data'].keys()))
n = len(demos)

np.random.seed(42)
perm = np.random.permutation(n)
n_train = int(0.9 * n)
train_demos = [demos[i] for i in perm[:n_train]]

for k in [5, 10, 20, 50]:
    mask_name = f'train_{k}'
    subset = train_demos[:k]
    if mask_name in f['mask']:
        del f['mask'][mask_name]
    f['mask'].create_dataset(mask_name, data=np.array(subset, dtype='S'))
    print(f"  {mask_name}: {len(subset)} demos")
f.close()

with open('configs/bc_image_tuneC.json') as fj:
    base = json.load(fj)

for k in [5, 10, 20, 50]:
    c = copy.deepcopy(base)
    c['experiment']['name'] = f'bc_demo_{k}'
    c['train']['output_dir'] = f'/home/ssenthilbabu/rbe577_p4/results/bc_demo_{k}'
    c['train']['hdf5_filter_key'] = f'train_{k}'
    c['train']['num_epochs'] = 300
    c['experiment']['rollout']['rate'] = 50
    c['experiment']['save']['every_n_epochs'] = 100
    with open(f'configs/bc_demo_{k}.json', 'w') as fj:
        json.dump(c, fj, indent=4)
    print(f"Config bc_demo_{k}.json written")
PYEOF

# Run only the demo count specified by DEMO_K env var
K=${DEMO_K}
mkdir -p ~/rbe577_p4/results/bc_demo_${K}
echo "=== Training with ${K} demos (300 epochs) ==="
yes | python robomimic/robomimic/scripts/train.py \
    --config configs/bc_demo_${K}.json \
    --dataset datasets/all_pickplace_image.hdf5 \
    --name bc_demo_${K} || echo "bc_demo_${K} FAILED"

echo "=== DONE: ${K} demos ==="
