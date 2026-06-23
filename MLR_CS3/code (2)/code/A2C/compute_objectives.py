import torch
import torch.nn.functional as F

def compute_discounted_returns(rewards, gamma, bootstrap_value=0.0):
    """G_t = r_t + gamma * G_{t+1} with bootstrapping for truncated episodes."""
    returns = []
    running_return = bootstrap_value
    
    for r in reversed(rewards):
        running_return = r + gamma * running_return
        returns.append(running_return)
        
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)

def compute_advantage(returns, values):
    """A_t = G_t - V(s_t). Values are detached to shield the Critic."""
    return returns - values.detach()

def normalize_advantage(advantages):
    """Crucial for stability in A2C."""
    if advantages.numel() <= 1: return advantages
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

def compute_actor_loss(log_probs, advantages):
    """L = -E[log_pi * A] (Negative for Gradient Ascent)"""
    return -(log_probs * advantages).mean()

def compute_critic_loss(returns, values):
    """L = MSE(G_t, V(s_t))"""
    return F.mse_loss(values, returns)