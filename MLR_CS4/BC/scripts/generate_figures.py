import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

figdir = '/home/shibuwu/Desktop/MLR/Project4/report/figures'

plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
})

BLUE = '#2171b5'
ORANGE = '#e6550d'
GREEN = '#31a354'
RED = '#de2d26'
PURPLE = '#756bb1'
TEAL = '#41b6c4'

def get_scalar(ea, tag):
    events = ea.Scalars(tag)
    return np.array([e.step for e in events]), np.array([e.value for e in events])

def ema(values, alpha=0.02):
    out = np.zeros_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i-1]
    return out

s = 20

# =========================================================================
# Image-based policy figures
# =========================================================================
img_base_ea = EventAccumulator('/home/shibuwu/Desktop/MLR/Project4/results/bc_image/logs/tb')
img_base_ea.Reload()
img_tuneA_ea = EventAccumulator('/home/shibuwu/Desktop/MLR/Project4/results/bc_image_tuneA/logs/tb')
img_tuneA_ea.Reload()
img_tuneB_ea = EventAccumulator('/home/shibuwu/Desktop/MLR/Project4/results/bc_image_tuneB/logs/tb')
img_tuneB_ea.Reload()
img_tuneC_ea = EventAccumulator('/home/shibuwu/Desktop/MLR/Project4/results/bc_image_tuneC/logs/tb')
img_tuneC_ea.Reload()

# Figure: Image BC Training Loss (all 4)
fig, ax = plt.subplots(figsize=(9, 5))
for ea, label, color in [
    (img_base_ea, 'Baseline (low-dim+img)', BLUE),
    (img_tuneA_ea, 'Tune A (smaller TF)', GREEN),
    (img_tuneB_ea, 'Tune B (batch=32, multistep)', PURPLE),
    (img_tuneC_ea, 'Tune C (images only)', ORANGE),
]:
    steps, vals = get_scalar(ea, 'Train/Loss')
    ax.plot(steps[s:], ema(vals, 0.01)[s:], color=color, label=label, linewidth=2.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Training Loss')
ax.set_title('Image-Based BC — Training Loss')
ax.legend()
ax.set_xlim(0, 3100)
plt.tight_layout()
plt.savefig(f'{figdir}/img_train_loss.png', dpi=200, bbox_inches='tight')
plt.close()
print('img_train_loss.png')

# Figure: Image BC Validation Loss (all 4)
fig, ax = plt.subplots(figsize=(9, 5))
for ea, label, color in [
    (img_base_ea, 'Baseline (low-dim+img)', BLUE),
    (img_tuneA_ea, 'Tune A (smaller TF)', GREEN),
    (img_tuneB_ea, 'Tune B (batch=32, multistep)', PURPLE),
    (img_tuneC_ea, 'Tune C (images only)', ORANGE),
]:
    steps, vals = get_scalar(ea, 'Valid/Loss')
    ax.plot(steps[s:], ema(vals, 0.01)[s:], color=color, label=label, linewidth=2.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Loss')
ax.set_title('Image-Based BC — Validation Loss')
ax.legend()
ax.set_xlim(0, 3100)
plt.tight_layout()
plt.savefig(f'{figdir}/img_valid_loss.png', dpi=200, bbox_inches='tight')
plt.close()
print('img_valid_loss.png')

# Figure: Image BC Success Rate (all 4)
fig, ax = plt.subplots(figsize=(9, 5))
for ea, label, color, marker in [
    (img_base_ea, 'Baseline (low-dim+img)', BLUE, 'o'),
    (img_tuneA_ea, 'Tune A (smaller TF)', GREEN, '^'),
    (img_tuneB_ea, 'Tune B (batch=32, multistep)', PURPLE, 'D'),
    (img_tuneC_ea, 'Tune C (images only)', ORANGE, 's'),
]:
    steps, vals = get_scalar(ea, 'Rollout/Success_Rate/all_pickplace_image')
    ax.plot(steps, vals * 100, f'{marker}-', color=color, label=label, markersize=8, linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Success Rate (%)')
ax.set_title('Image-Based BC — Rollout Success Rate')
ax.legend()
ax.set_ylim(-5, 110)
ax.set_xlim(0, 2200)
plt.tight_layout()
plt.savefig(f'{figdir}/img_success_rate.png', dpi=200, bbox_inches='tight')
plt.close()
print('img_success_rate.png')

# Figure: Image BC bar chart — Peak Success
fig, ax = plt.subplots(figsize=(8, 5))
names_img = ['Baseline\n(low-dim+img)', 'Tune A\n(smaller TF)', 'Tune B\n(batch=32\nmultistep)', 'Tune C\n(images only)']
vals_img = [90, 90, 90, 100]
colors_img = [BLUE, GREEN, PURPLE, ORANGE]
bars = ax.bar(names_img, vals_img, color=colors_img, edgecolor='white', linewidth=1.5, width=0.7)
for bar, val in zip(bars, vals_img):
    y = bar.get_height() + 1.5
    ax.text(bar.get_x() + bar.get_width() / 2, y, f'{val}%',
            ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.set_ylabel('Peak Success Rate (%)')
ax.set_title('Image-Based BC — Peak Success Rate')
ax.set_ylim(0, 115)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{figdir}/img_experiments_bar.png', dpi=200, bbox_inches='tight')
plt.close()
print('img_experiments_bar.png')

# =========================================================================
# Figure 8: Environment image from rollout video
# =========================================================================
import imageio
reader = imageio.get_reader('/home/shibuwu/Desktop/MLR/Project4/submission/videos/bc_best_rollout.mp4')
frame = reader.get_data(30)
reader.close()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
h, w = frame.shape[:2]
ax1.imshow(frame[:, :w//2, :])
ax1.set_title('Agent View', fontsize=13)
ax1.axis('off')
ax2.imshow(frame[:, w//2:, :])
ax2.set_title('Wrist Camera View', fontsize=13)
ax2.axis('off')
fig.suptitle('PickPlaceCan — Panda Robot', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(f'{figdir}/pickplace_env.png', dpi=200, bbox_inches='tight')
plt.close()
print('8/8 pickplace_env.png')

print('\nAll figures done!')
