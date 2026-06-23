import torch
import torch.nn as nn
from torch.distributions import Categorical

HIDDEN_DIM = 512   # single source of truth — must match train.py and critic.py


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=HIDDEN_DIM):
        super(Actor, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Initialise all linear layers with gain=1.0
        for m in self.nn:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0)

        # Output layer: tiny gain → near-uniform initial action probs (~25% each)
        # MUST be done after the loop, not inside it (m == self.nn[-1] is unreliable)
        nn.init.orthogonal_(self.nn[-1].weight, gain=0.01)
        nn.init.constant_(self.nn[-1].bias, 0)

    def forward(self, x):
        return self.nn(x)

    def get_action(self, state, deterministic=False):
        if state.ndim == 1:
            state = state.unsqueeze(0)
        logits = self.forward(state)
        if deterministic:
            return torch.argmax(logits, dim=-1).item()
        return Categorical(logits=logits).sample().item()

    def evaluate_actions(self, state, action):
        if state.ndim == 1:
            state = state.unsqueeze(0)
        dist      = Categorical(logits=self.forward(state))
        log_probs = dist.log_prob(action)
        entropy   = dist.entropy()
        return log_probs, entropy
