import os
import sys
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import torch
import torch.nn.functional as F

from helpers.metrics import MetricsTracker
from helpers.utils import get_network_input_shape, get_screen, make_env, setup_camera
from lib.a3c.model import ActorCritic
from lib.a3c.objectives import (
    compute_actor_loss,
    compute_advantage,
    compute_bootstrapped_returns,
    compute_critic_loss,
)


def emit_log(message, log_path=None):
    print(message, flush=True)
    if log_path is None:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"{timestamp} - INFO - {message}\n")


def worker_process(
    worker_id,
    global_net,
    optimizer,
    global_ep,
    max_episodes,
    lock,
    config,
    device,
    shared_stats,
    log_path=None,
):
    env = make_env(config, worker_id)

    local_net = ActorCritic(
        state_size=config["network"]["state_size"],
        action_size=env.action_space.shape[0],
        shared_layers=config["network"]["shared_layers"],
        critic_hidden_layers=config["network"]["critic_hidden_layers"],
        actor_hidden_layers=config["network"]["actor_hidden_layers"],
        init_type=config["network"]["init_type"],
        seed=worker_id,
    ).to(device)
    local_net.load_state_dict(global_net.state_dict())

    metrics = MetricsTracker()

    gamma = config["hyperparameters"]["gamma"]
    t_max = config["hyperparameters"]["t_max"]
    entropy_coef = config["hyperparameters"]["entropy_coef"]
    value_loss_coef = config["hyperparameters"]["value_loss_coef"]
    grad_clip = config["hyperparameters"]["grad_clip"]
    log_interval = config["logging"]["log_interval"]
    save_interval = config["logging"]["save_interval"]

    env.reset()
    setup_camera(env, config)
    state = get_screen(env, device, config)
    episode_reward = 0.0
    episode_steps = 0

    while True:
        with global_ep.get_lock():
            if global_ep.value >= max_episodes:
                break

        local_net.load_state_dict(global_net.state_dict())
        local_net.zero_grad()

        log_probs = []
        values = []
        rewards = []
        entropies = []
        done = False

        for _ in range(t_max):
            action_loc, value = local_net(state)
            dist = local_net.get_action_distribution(action_loc)

            action = dist.rsample()
            log_prob = dist.log_prob(action)
            entropy = dist.base_dist_for_entropy.entropy()
            action_np = action.squeeze(0).detach().cpu().numpy()

            _, reward, done, _ = env.step(action_np)
            next_state = get_screen(env, device, config)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            entropies.append(entropy)

            episode_reward += reward
            episode_steps += 1
            state = next_state

            if done:
                break

        with torch.no_grad():
            if done:
                bootstrap_value = 0.0
            else:
                _, v = local_net(state)
                bootstrap_value = v.item()

        return_batch = compute_bootstrapped_returns(rewards, gamma, bootstrap_value).to(device)
        log_prob_batch = torch.stack(log_probs).squeeze()
        value_batch = torch.stack(values).squeeze()
        entropy_batch = torch.stack(entropies).squeeze()
        advantage_batch = compute_advantage(return_batch, value_batch)
        actor_loss = compute_actor_loss(log_prob_batch, advantage_batch, entropy_batch, entropy_coef)
        critic_loss = compute_critic_loss(return_batch, value_batch)
        total_loss = actor_loss + value_loss_coef * critic_loss

        actor_loss_value = actor_loss.item()
        critic_loss_value = critic_loss.item()
        total_loss_value = total_loss.item()
        policy_std_value = float(F.softplus(local_net.sigma).mean().item())

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(local_net.parameters(), grad_clip)

        with lock:
            for local_param, global_param in zip(local_net.parameters(), global_net.parameters()):
                if global_param.grad is None:
                    global_param.grad = local_param.grad
                else:
                    global_param.grad = local_param.grad
            optimizer.step()

        current_ep = None
        if done:
            metrics.add_episode_reward(episode_reward)
            metrics.add_loss(total_loss_value)
            metrics.add_episode_length(episode_steps)

            with global_ep.get_lock():
                global_ep.value += 1
                current_ep = global_ep.value

            shared_stats["rewards"].append(episode_reward)
            shared_stats["lengths"].append(episode_steps)
            shared_stats["losses"].append(total_loss_value)

            if current_ep % log_interval == 0:
                avg_reward = np.mean(list(shared_stats["rewards"])[-log_interval:])
                emit_log(
                    f"Worker {worker_id} | Episode {current_ep} | "
                    f"Reward: {episode_reward:.2f} | Avg: {avg_reward:.2f} | "
                    f"Actor Loss: {actor_loss_value:.4f} | Critic Loss: {critic_loss_value:.4f} | "
                    f"Sigma: {policy_std_value:.4f}",
                    log_path,
                )

            if current_ep % save_interval == 0:
                model_dir = config["logging"]["model_dir"]
                os.makedirs(model_dir, exist_ok=True)
                torch.save(
                    {"model_state_dict": global_net.state_dict(), "episode": current_ep},
                    os.path.join(model_dir, f"a3c_kuka_ep{current_ep}.pth"),
                )

            env.reset()
            setup_camera(env, config)
            state = get_screen(env, device, config)
            episode_reward = 0.0
            episode_steps = 0

        if current_ep is not None and current_ep >= max_episodes:
            break

    env.close()
