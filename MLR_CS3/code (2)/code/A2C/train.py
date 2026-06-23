import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import imageio.v2 as imageio
import gymnasium as gym
import matplotlib.pyplot as plt
from actor import Actor
from critic import Critic
from compute_objectives import (
    compute_discounted_returns,
    compute_advantage,
    normalize_advantage,
    compute_actor_loss,
    compute_critic_loss,
)
from utils import load_config, set_random_seed, ObservationNormalizer, VIDEOS_DIR

# ── Single source of truth for network size ───────────────────────────────────
HIDDEN_DIM = 512   # must match actor.py and critic.py


def run_lunar_lander(actor, video_filename="lunar_lander_best.mp4", config=None):
    """Record one deterministic episode for the deliverable video."""
    config = config or load_config()
    env    = gym.make(config["env_id"], render_mode="rgb_array")
    raw_state, _ = env.reset()
    obs_normalizer = getattr(actor, "obs_normalizer", None)
    frames = []

    for _ in range(config["max_ep_steps"]):
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        state_norm   = obs_normalizer.normalize(raw_state)
        state_tensor = torch.tensor(state_norm, dtype=torch.float32).to(
            next(actor.parameters()).device
        )
        with torch.no_grad():
            action = actor.get_action(state_tensor, deterministic=True)

        raw_state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    env.close()
    if frames:
        os.makedirs(VIDEOS_DIR, exist_ok=True)
        path = os.path.join(VIDEOS_DIR, video_filename)
        imageio.mimsave(path, frames, fps=30)
        print(f">>> Video saved to {path}")


def train_actor_critic(config_path=None):
    config = load_config(config_path) if config_path else load_config()
    set_random_seed(config["random_seed"])

    # Save hidden_dim into config so eval.py can load the right architecture
    config["hidden_dim"] = HIDDEN_DIM
    config["state_dim"]  = 8
    config["action_dim"] = 4

    device = torch.device("cpu")
    print(f">>> TRAINING A2C ON: {device} <<<", flush=True)

    env        = gym.make(config["env_id"])
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor  = Actor(state_dim, action_dim, HIDDEN_DIM).to(device)
    critic = Critic(state_dim, HIDDEN_DIM).to(device)

    obs_normalizer = ObservationNormalizer(state_dim)
    actor.obs_normalizer = obs_normalizer

    actor_optim  = optim.Adam(actor.parameters(),  lr=5e-4)
    critic_optim = optim.Adam(critic.parameters(), lr=5e-4)

    actor_scheduler  = optim.lr_scheduler.StepLR(actor_optim,  step_size=1000, gamma=0.5)
    critic_scheduler = optim.lr_scheduler.StepLR(critic_optim, step_size=1000, gamma=0.5)

    entropy_coeff     = 0.15
    entropy_decay     = 0.9998
    min_entropy_coeff = 0.01

    reward_history  = []
    best_avg_reward = -float("inf")
    total_episodes  = 5000

    os.makedirs("./checkpoints", exist_ok=True)

    for i_episode in range(total_episodes):

        raw_state, _ = env.reset()
        state = torch.tensor(
            obs_normalizer.normalize(raw_state), dtype=torch.float32
        ).to(device)

        episode_reward  = 0.0
        episode_states  = []
        episode_actions = []
        episode_rewards = []

        for _ in range(config["max_ep_steps"]):
            action = actor.get_action(state, deterministic=False)
            next_raw_state, reward, terminated, truncated, _ = env.step(action)

            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_reward += reward

            obs_normalizer.update(next_raw_state)
            state = torch.tensor(
                obs_normalizer.normalize(next_raw_state), dtype=torch.float32
            ).to(device)

            if terminated or truncated:
                break

        reward_history.append(episode_reward)

        state_batch  = torch.stack(episode_states).to(device)
        action_batch = torch.tensor(episode_actions, dtype=torch.int64).to(device)

        bootstrap_val = 0.0
        if truncated:
            with torch.no_grad():
                bootstrap_val = critic(state).item()

        returns    = compute_discounted_returns(
            episode_rewards, gamma=0.99, bootstrap_value=bootstrap_val
        ).to(device)

        log_probs, entropy = actor.evaluate_actions(state_batch, action_batch)
        values             = critic(state_batch)

        critic_loss = compute_critic_loss(returns, values)
        advantages  = normalize_advantage(compute_advantage(returns, values))
        actor_loss  = compute_actor_loss(log_probs, advantages)

        total_loss  = actor_loss + 0.5 * critic_loss - entropy_coeff * entropy.mean()

        actor_optim.zero_grad()
        critic_optim.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(),  0.5)
        nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
        actor_optim.step()
        critic_optim.step()

        entropy_coeff = max(min_entropy_coeff, entropy_coeff * entropy_decay)

        if (i_episode + 1) > 2000:
            actor_scheduler.step()
            critic_scheduler.step()

        if (i_episode + 1) % 100 == 0:
            avg_r    = np.mean(reward_history[-100:])
            actor_lr = actor_optim.param_groups[0]["lr"]

            if avg_r > best_avg_reward:
                best_avg_reward = avg_r
                torch.save(
                    {
                        "actor_state_dict":     actor.state_dict(),
                        "obs_normalizer_state": obs_normalizer.state_dict(),
                        "config":               config,
                    },
                    "./checkpoints/best_actor_critic.pt",
                )
                print(
                    f"Ep {i_episode+1:>5} | Avg Reward: {avg_r:>8.2f} | "
                    f"Loss: {total_loss.item():>10.4f} | "
                    f"LR: {actor_lr:.2e} | "
                    f"Ent: {entropy_coeff:.4f}  --> new best, saved",
                    flush=True,
                )
            else:
                print(
                    f"Ep {i_episode+1:>5} | Avg Reward: {avg_r:>8.2f} | "
                    f"Loss: {total_loss.item():>10.4f} | "
                    f"LR: {actor_lr:.2e} | "
                    f"Ent: {entropy_coeff:.4f}",
                    flush=True,
                )

    # ── PLOT ──────────────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, label='Raw Reward', alpha=0.3)
    if len(reward_history) >= 100:
        ma = np.convolve(reward_history, np.ones(100) / 100, mode='valid')
        plt.plot(range(99, len(reward_history)), ma,
                 label='100-Ep Moving Avg', color='red')
    plt.title("LunarLander-v2 Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig("reward_plot.png")
    plt.close()
    print(">>> Training plot saved to reward_plot.png")

    # ── SAVE FINAL WEIGHTS ────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(config["checkpoint_path"]), exist_ok=True)
    torch.save(
        {
            "actor_state_dict":     actor.state_dict(),
            "obs_normalizer_state": obs_normalizer.state_dict(),
            "config":               config,
        },
        config["checkpoint_path"],
    )
    print(f">>> Final weights saved to {config['checkpoint_path']}")
    return actor


if __name__ == "__main__":
    train_actor_critic()

    config = load_config()
    checkpoint = torch.load(
        "./checkpoints/best_actor_critic.pt", weights_only=False
    )

    saved_config = checkpoint.get("config", config)
    hidden_dim   = saved_config.get("hidden_dim", HIDDEN_DIM)
    state_dim    = saved_config.get("state_dim",  8)
    action_dim   = saved_config.get("action_dim", 4)

    best_actor = Actor(state_dim, action_dim, hidden_dim)
    best_actor.load_state_dict(checkpoint["actor_state_dict"])

    obs_normalizer = ObservationNormalizer(state_dim)
    obs_normalizer.load_state_dict(checkpoint["obs_normalizer_state"])
    best_actor.obs_normalizer = obs_normalizer

    run_lunar_lander(best_actor, video_filename="lunar_lander_best.mp4")