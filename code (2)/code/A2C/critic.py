import torch
import torch.nn as nn

HIDDEN_DIM = 512   # must match actor.py and train.py


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=HIDDEN_DIM):
        super(Critic, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)   # NO ReLU — value estimates are unbounded
        )

        # Initialise all layers with gain=1.0
        self.apply(self._init_weights)

        # Output layer: small gain keeps initial value estimates near zero
        nn.init.orthogonal_(self.nn[-1].weight, gain=0.1)
        nn.init.constant_(self.nn[-1].bias, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, state):
        if state.ndim == 1:
            state = state.unsqueeze(0)
        return self.nn(state).squeeze(-1)
