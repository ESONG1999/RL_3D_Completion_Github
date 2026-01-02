# src/models/rl_scheduler.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch import Tensor


@dataclass
class RLSchedulerConfig:
    actions: List[int]  # allowed skip lengths, e.g. [1,2,4]
    hidden_dim: int = 128
    state_dim: int = 64  # simple embedding dim for state


class RLScheduler(nn.Module):
    """
    Simple REINFORCE policy for choosing skip lengths.
    State features are minimal: normalized timestep + rough global stats.
    """

    def __init__(self, cfg: RLSchedulerConfig):
        super().__init__()
        self.cfg = cfg
        A = len(cfg.actions)

        self.net = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, A),
        )

    def forward(self, state: Tensor) -> Tensor:
        """
        state: (B, state_dim)
        returns logits: (B, A)
        """
        return self.net(state)

    def select_action(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Returns:
          - actions indices: (B,)
          - log_probs: (B,)
        """
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        return action_idx, log_prob
