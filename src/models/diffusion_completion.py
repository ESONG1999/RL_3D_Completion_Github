# src/models/diffusion_completion.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class DiffusionConfig:
    num_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02


def sinusoidal_time_embedding(timesteps: Tensor, dim: int) -> Tensor:
    """
    timesteps: (B,)
    returns: (B, dim)
    """
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(
            math.log(1.0),
            math.log(10000.0),
            steps=half,
            device=device,
        )
        * (-1.0 / (half - 1))
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class PointNetEncoder(nn.Module):
    """Simple PointNet-style encoder for partial point cloud."""

    def __init__(self, in_dim: int = 3, hidden_dim: int = 128, out_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, Np, 3)
        returns: (B, out_dim)
        """
        feat = self.mlp(x)  # (B, Np, out_dim)
        feat = feat.max(dim=1).values  # global max pool
        return feat


class Denoiser(nn.Module):
    """Per-point denoiser with conditioning."""

    def __init__(self, in_dim: int = 3, cond_dim: int = 256, time_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.fc_in = nn.Linear(in_dim + cond_dim + time_dim, hidden_dim)
        self.block1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.block2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fc_out = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x_t: Tensor, cond: Tensor, t_emb: Tensor) -> Tensor:
        """
        x_t: (B, N, 3)
        cond: (B, C)
        t_emb: (B, Tdim)
        """
        B, N, _ = x_t.shape
        cond_exp = cond.unsqueeze(1).expand(B, N, cond.shape[-1])
        t_exp = t_emb.unsqueeze(1).expand(B, N, t_emb.shape[-1])
        h = torch.cat([x_t, cond_exp, t_exp], dim=-1)
        h = self.fc_in(h)
        h = h + self.block1(h)
        h = h + self.block2(h)
        out = self.fc_out(h)
        return out


class DiffusionCompletionModel(nn.Module):
    def __init__(
        self,
        cfg: DiffusionConfig,
        cond_dim: int = 256,
        time_dim: int = 128,
    ):
        super().__init__()
        self.cfg = cfg
        self.encoder = PointNetEncoder(in_dim=3, hidden_dim=128, out_dim=cond_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.ReLU(inplace=True),
            nn.Linear(time_dim, time_dim),
        )
        self.denoiser = Denoiser(
            in_dim=3,
            cond_dim=cond_dim,
            time_dim=time_dim,
            hidden_dim=256,
        )
        self.register_buffer("betas", self._make_beta_schedule(cfg), persistent=False)
        self._build_alphas()

    def _make_beta_schedule(self, cfg: DiffusionConfig) -> Tensor:
        return torch.linspace(cfg.beta_start, cfg.beta_end, cfg.num_steps)

    def _build_alphas(self):
        betas = self.betas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)

    def q_sample(self, x0: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        """
        x0: (B, N, 3)
        t: (B,) timestep indices in [0, T-1]
        """
        if noise is None:
            noise = torch.randn_like(x0)
        alphas_cumprod = self.alphas_cumprod[t].view(-1, 1, 1)  # (B,1,1)
        return torch.sqrt(alphas_cumprod) * x0 + torch.sqrt(1 - alphas_cumprod) * noise

    def forward(self, full_points: Tensor, partial_points: Tensor, t: Tensor) -> Tensor:
        """
        Training forward: predict noise for x_t at timestep t.
        """
        cond = self.encoder(partial_points)  # (B, C)
        t_emb = sinusoidal_time_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)

        noise = torch.randn_like(full_points)
        x_t = self.q_sample(full_points, t, noise)
        noise_pred = self.denoiser(x_t, cond, t_emb)
        loss = torch.mean((noise - noise_pred) ** 2)
        return loss

    @torch.no_grad()
    def ddim_sample(
        self,
        partial_points: Tensor,
        num_steps: int = 50,
        eta: float = 0.0,
        device: Optional[torch.device] = None,
        schedule: Optional[list[int]] = None,
    ) -> Tensor:
        """
        DDIM sampling with optional custom schedule of timesteps.
        Returns: x0: (B, N, 3)
        """
        self.eval()
        B = partial_points.shape[0]
        device = device or partial_points.device

        cond = self.encoder(partial_points.to(device))
        time_dim = self.time_mlp[0].in_features

        T = self.cfg.num_steps
        if schedule is None:
            # uniform from T-1 to 0
            schedule = torch.linspace(T - 1, 0, num_steps, dtype=torch.long)
        else:
            schedule = torch.tensor(schedule, dtype=torch.long)
        schedule = schedule.to(device)

        x = torch.randn(B, partial_points.shape[1], 3, device=device)

        for i in range(len(schedule) - 1):
            t = schedule[i].expand(B)
            t_prev = schedule[i + 1].expand(B)

            t_emb = sinusoidal_time_embedding(t, time_dim).to(device)
            t_emb = self.time_mlp(t_emb)
            eps = self.denoiser(x, cond, t_emb)

            a_t = self.alphas_cumprod[t].view(-1, 1, 1)
            a_prev = self.alphas_cumprod[t_prev].view(-1, 1, 1)
            x0_pred = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t + 1e-8)

            if eta == 0.0:
                x = torch.sqrt(a_prev) * x0_pred + torch.sqrt(1 - a_prev) * eps
            else:
                sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
                noise = torch.randn_like(x)
                x = torch.sqrt(a_prev) * x0_pred + torch.sqrt(1 - a_prev - sigma**2) * eps + sigma * noise

        # final step to t=0
        t0 = schedule[-1].expand(B)
        t_emb0 = sinusoidal_time_embedding(t0, time_dim).to(device)
        t_emb0 = self.time_mlp(t_emb0)
        eps0 = self.denoiser(x, cond, t_emb0)
        a0 = self.alphas_cumprod[t0].view(-1, 1, 1)
        x0 = (x - torch.sqrt(1 - a0) * eps0) / torch.sqrt(a0 + 1e-8)
        return x0
