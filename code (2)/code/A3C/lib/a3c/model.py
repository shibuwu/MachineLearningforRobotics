import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal, TanhTransform, TransformedDistribution
from helpers.utils import build_hidden_layer


class ActorCritic(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        shared_layers,
        critic_hidden_layers=None,
        actor_hidden_layers=None,
        seed=0,
        init_type=None,
    ):
        super(ActorCritic, self).__init__()
        self.init_type = init_type
        torch.manual_seed(seed)

        # Learned std dev for the policy; negative init so softplus starts small (~0.31)
        self.sigma = nn.Parameter(torch.full((action_size,), -1.0))

        critic_hidden_layers = critic_hidden_layers or []
        actor_hidden_layers = actor_hidden_layers or []

        # Convolutional encoder for image observations
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d(2)

        linear_input_size = 64 * 2 * 2  # 256

        self.shared_layers = build_hidden_layer(linear_input_size, shared_layers)

        # Critic head
        last_shared = shared_layers[-1] if shared_layers else linear_input_size
        if critic_hidden_layers:
            self.critic_hidden = build_hidden_layer(last_shared, critic_hidden_layers)
            self.critic = nn.Linear(critic_hidden_layers[-1], 1)
        else:
            self.critic_hidden = None
            self.critic = nn.Linear(last_shared, 1)

        # Actor head
        if actor_hidden_layers:
            self.actor_hidden = build_hidden_layer(last_shared, actor_hidden_layers)
            self.actor = nn.Linear(actor_hidden_layers[-1], action_size)
        else:
            self.actor_hidden = None
            self.actor = nn.Linear(last_shared, action_size)

        if self.init_type is not None:
            self.shared_layers.apply(self._initialize)
            self.critic.apply(self._initialize)
            self.actor.apply(self._initialize)
            if self.critic_hidden is not None:
                self.critic_hidden.apply(self._initialize)
            if self.actor_hidden is not None:
                self.actor_hidden.apply(self._initialize)

    def _initialize(self, n):
        if isinstance(n, nn.Linear):
            if self.init_type == "xavier-uniform":
                nn.init.xavier_uniform_(n.weight.data)
            elif self.init_type == "xavier-normal":
                nn.init.xavier_normal_(n.weight.data)
            elif self.init_type == "kaiming-uniform":
                nn.init.kaiming_uniform_(n.weight.data)
            elif self.init_type == "kaiming-normal":
                nn.init.kaiming_normal_(n.weight.data)
            elif self.init_type == "orthogonal":
                nn.init.orthogonal_(n.weight.data)
            elif self.init_type == "uniform":
                nn.init.uniform_(n.weight.data)
            elif self.init_type == "normal":
                nn.init.normal_(n.weight.data)
            else:
                raise KeyError("initialization type not found")

    def forward(self, state):
        state = (state - 0.5) / 0.5

        x = self.encoder(state)
        x = self.spatial_pool(x)
        x = x.view(x.size(0), -1)
        for layer in self.shared_layers:
            x = F.relu(layer(x))

        v = x
        if self.critic_hidden is not None:
            for layer in self.critic_hidden:
                v = F.relu(layer(v))
        value = self.critic(v)

        a = x
        if self.actor_hidden is not None:
            for layer in self.actor_hidden:
                a = F.relu(layer(a))
        action_loc = self.actor(a)

        return action_loc, value

    def get_action_distribution(self, action_loc):
        """Tanh-squashed Gaussian over bounded actions."""
        sigma = F.softplus(self.sigma)
        base_dist = Independent(Normal(action_loc, sigma), 1)
        dist = TransformedDistribution(base_dist, [TanhTransform(cache_size=1)])
        # TransformedDistribution doesn't support .entropy(), keep ref to base
        dist.base_dist_for_entropy = base_dist
        return dist
