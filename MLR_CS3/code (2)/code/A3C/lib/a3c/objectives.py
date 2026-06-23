import torch
import torch.nn.functional as F


def compute_bootstrapped_returns(rewards, gamma, bootstrap_value):
    """Discounted returns with bootstrap at rollout boundary."""
    running_return = bootstrap_value
    returns = []
    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.insert(0, running_return)
    return torch.tensor(returns, dtype=torch.float32)


def compute_advantage(return_batch, value_batch):
    return return_batch - value_batch.detach()


def compute_actor_loss(log_prob_batch, advantage_batch, entropy_batch, entropy_coef):
    policy_loss = -(log_prob_batch * advantage_batch.detach()).mean()
    entropy_bonus = entropy_batch.mean()
    return policy_loss - entropy_coef * entropy_bonus


def compute_critic_loss(return_batch, value_batch):
    return F.mse_loss(value_batch.view(-1), return_batch.view(-1))

