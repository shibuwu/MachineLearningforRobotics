import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_log(path):
    episodes, rewards, actor_losses, critic_losses, sigmas = [], [], [], [], []
    with open(path) as f:
        for line in f:
            m = re.search(
                r'Episode (\d+) \| Reward: ([\d.]+) \| Avg: ([\d.]+) \| '
                r'Actor Loss: ([-.e\d]+) \| Critic Loss: ([-.e\d]+) \| Sigma: ([-.e\d]+)',
                line
            )
            if m:
                episodes.append(int(m.group(1)))
                rewards.append(float(m.group(3)))  # use Avg reward
                actor_losses.append(float(m.group(4)))
                critic_losses.append(float(m.group(5)))
                sigmas.append(float(m.group(6)))
    return np.array(episodes), np.array(rewards), np.array(actor_losses), np.array(critic_losses), np.array(sigmas)

# Parse the retrain log
eps, avg_rwd, actor_loss, critic_loss, sigma = parse_log('logs/v6_train.log')

# Smooth with rolling window
def smooth(x, w=50):
    return np.convolve(x, np.ones(w)/w, mode='valid')

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('A3C Training Curves (V6 Config, 15k Episodes)', fontsize=14)

# Avg reward
ax = axes[0, 0]
ax.plot(eps, avg_rwd, alpha=0.2, color='blue')
sm_eps = eps[len(eps)-len(smooth(avg_rwd)):]
ax.plot(sm_eps, smooth(avg_rwd), color='blue', linewidth=1.5)
ax.set_xlabel('Episode')
ax.set_ylabel('Avg Reward (10-ep window)')
ax.set_title('Average Reward')
ax.set_ylim(-0.05, 1.05)
ax.axhline(y=0.25, color='red', linestyle='--', alpha=0.5, label='25% target')
ax.legend()

# Actor loss
ax = axes[0, 1]
ax.plot(eps, actor_loss, alpha=0.2, color='orange')
ax.plot(sm_eps, smooth(actor_loss), color='orange', linewidth=1.5)
ax.set_xlabel('Episode')
ax.set_ylabel('Actor Loss')
ax.set_title('Actor Loss')

# Critic loss
ax = axes[1, 0]
ax.plot(eps, critic_loss, alpha=0.2, color='green')
ax.plot(sm_eps, smooth(critic_loss), color='green', linewidth=1.5)
ax.set_xlabel('Episode')
ax.set_ylabel('Critic Loss')
ax.set_title('Critic Loss')

# Sigma
ax = axes[1, 1]
ax.plot(eps, sigma, color='purple', linewidth=1.5)
ax.set_xlabel('Episode')
ax.set_ylabel('Sigma (softplus)')
ax.set_title('Policy Std Dev')

plt.tight_layout()
import os
os.makedirs('/home/shibuwu/Desktop/MLR/Project3/MLR_P3/figures', exist_ok=True)
plt.savefig('/home/shibuwu/Desktop/MLR/Project3/MLR_P3/figures/a3c_training_curves.png', dpi=150, bbox_inches='tight')
print('Saved to MLR_P3/figures/a3c_training_curves.png')
