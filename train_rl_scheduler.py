#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.utils.seed import set_seed
from src.utils.device import get_device
from src.utils.chamfer import chamfer_distance
from src.models.diffusion_completion import (
    DiffusionCompletionModel,
    DiffusionConfig,
    sinusoidal_time_embedding,
)
from src.models.rl_scheduler import RLScheduler, RLSchedulerConfig


class CompletionDataset(Dataset):
    def __init__(self, npz_path: str):
        arr = np.load(npz_path)
        self.full = arr["full_points"].astype("float32")
        self.partial = arr["partial_points"].astype("float32")

    def __len__(self):
        return self.full.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.full[idx]),
            torch.from_numpy(self.partial[idx]),
        )


def build_state(x_t: torch.Tensor, t: int, T: int) -> torch.Tensor:
    B, N, _ = x_t.shape
    assert B == 1
    feat = []

    feat.append(torch.tensor([t / float(T)], device=x_t.device))
    mean = x_t.mean(dim=(0, 1))
    var = x_t.var(dim=(0, 1))
    feat.append(mean)
    feat.append(var)

    f = torch.cat(feat, dim=0)  # 1 + 3 + 3 = 7
    state_dim = 64
    if f.numel() < state_dim:
        f = torch.cat(
            [f, torch.zeros(state_dim - f.numel(), device=x_t.device)],
            dim=0,
        )
    else:
        f = f[:state_dim]
    return f.unsqueeze(0)  # (1, state_dim)


@torch.no_grad()
def rollout_episode(
    diffusion: DiffusionCompletionModel,
    policy: RLScheduler,
    full: torch.Tensor,
    partial: torch.Tensor,
    T: int,
    rl_cfg: RLSchedulerConfig,
    lambda_steps: float,
    device: torch.device,
):
    diffusion.eval()
    policy.eval()

    full = full.unsqueeze(0).to(device)      # (1, N, 3)
    partial = partial.unsqueeze(0).to(device)  # (1, Np, 3)

    x = torch.randn_like(full)
    t = T - 1
    steps = 0

    # 预先计算 condition
    cond = diffusion.encoder(partial)

    while t > 0 and steps < rl_cfg.max_steps_per_episode:
        state = build_state(x, t, T)  # (1, state_dim)
        action_idx, _ = policy.select_action(state)
        skip = rl_cfg.actions[action_idx.item()]

        t_tensor = torch.tensor([t], device=device, dtype=torch.long)
        t_prev = max(t - skip, 0)
        t_prev_tensor = torch.tensor([t_prev], device=device, dtype=torch.long)

        time_dim = diffusion.time_mlp[0].in_features
        t_emb = sinusoidal_time_embedding(t_tensor, time_dim)
        t_emb = diffusion.time_mlp(t_emb)

        eps = diffusion.denoiser(x, cond, t_emb)
        a_t = diffusion.alphas_cumprod[t_tensor].view(-1, 1, 1)
        a_prev = diffusion.alphas_cumprod[t_prev_tensor].view(-1, 1, 1)
        x0_pred = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t + 1e-8)
        x = torch.sqrt(a_prev) * x0_pred + torch.sqrt(1 - a_prev) * eps

        t = t_prev
        steps += 1
        if t == 0:
            break

    # final denoise at t=0
    t_tensor = torch.tensor([0], device=device, dtype=torch.long)
    time_dim = diffusion.time_mlp[0].in_features
    t_emb0 = sinusoidal_time_embedding(t_tensor, time_dim)
    t_emb0 = diffusion.time_mlp(t_emb0)
    eps0 = diffusion.denoiser(x, cond, t_emb0)
    a0 = diffusion.alphas_cumprod[t_tensor].view(-1, 1, 1)
    x0 = (x - torch.sqrt(1 - a0) * eps0) / torch.sqrt(a0 + 1e-8)

    cd = chamfer_distance(x0, full).item()
    reward = -cd - lambda_steps * steps

    return cd, reward, steps


def train(args):
    set_seed(args.seed)
    device = get_device(args.device)

    diff_cfg = DiffusionConfig(num_steps=args.num_steps)
    diffusion = DiffusionCompletionModel(diff_cfg).to(device)
    diffusion.load_state_dict(torch.load(args.ckpt, map_location=device))
    diffusion.eval()

    rl_cfg = RLSchedulerConfig(actions=args.actions)
    rl_cfg.max_steps_per_episode = args.max_steps

    policy = RLScheduler(rl_cfg).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    train_ds = CompletionDataset(args.train_npz)
    val_ds = CompletionDataset(args.val_npz)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    train_iter = iter(train_loader)

    workdir = Path(args.workdir)
    (workdir / "checkpoints").mkdir(parents=True, exist_ok=True)

    T = args.num_steps
    baseline = 0.0
    alpha = 0.9  # moving average baseline

    best_val_reward = -1e9

    for episode in range(1, args.episodes + 1):
        try:
            full, partial = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            full, partial = next(train_iter)

        full = full.to(device)      # (1, N, 3)
        partial = partial.to(device)  # (1, Np, 3)

        diffusion.eval()
        policy.train()

        x = torch.randn_like(full)
        t = T - 1

        log_probs = []
        steps = 0

        with torch.no_grad():
            cond = diffusion.encoder(partial)

        while t > 0 and steps < args.max_steps:
            state = build_state(x, t, T)  # (1, state_dim)
            action_idx, log_prob = policy.select_action(state)
            skip = rl_cfg.actions[action_idx.item()]
            log_probs.append(log_prob)

            t_tensor = torch.tensor([t], device=device, dtype=torch.long)
            t_prev = max(t - skip, 0)
            t_prev_tensor = torch.tensor([t_prev], device=device, dtype=torch.long)

            time_dim = diffusion.time_mlp[0].in_features
            with torch.no_grad():
                t_emb = sinusoidal_time_embedding(t_tensor, time_dim)
                t_emb = diffusion.time_mlp(t_emb)
                eps = diffusion.denoiser(x, cond, t_emb)
                a_t = diffusion.alphas_cumprod[t_tensor].view(-1, 1, 1)
                a_prev = diffusion.alphas_cumprod[t_prev_tensor].view(-1, 1, 1)
                x0_pred = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t + 1e-8)
                x = torch.sqrt(a_prev) * x0_pred + torch.sqrt(1 - a_prev) * eps

            t = t_prev
            steps += 1
            if t == 0:
                break

        # final denoise at t=0
        with torch.no_grad():
            t_tensor = torch.tensor([0], device=device, dtype=torch.long)
            time_dim = diffusion.time_mlp[0].in_features
            t_emb0 = sinusoidal_time_embedding(t_tensor, time_dim)
            t_emb0 = diffusion.time_mlp(t_emb0)
            eps0 = diffusion.denoiser(x, cond, t_emb0)
            a0 = diffusion.alphas_cumprod[t_tensor].view(-1, 1, 1)
            x0 = (x - torch.sqrt(1 - a0) * eps0) / torch.sqrt(a0 + 1e-8)

            cd = chamfer_distance(x0, full).item()

        reward = -cd - args.lambda_steps * steps

        # moving baseline
        baseline = alpha * baseline + (1 - alpha) * reward
        advantage = reward - baseline

        # policy gradient
        loss = 0.0
        for lp in log_probs:
            loss = loss - lp * advantage

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % args.log_interval == 0:
            print(
                f"[Train][Episode {episode}] "
                f"reward={reward:.4f} cd={cd:.4f} steps={steps} baseline={baseline:.4f}"
            )

        if episode % args.eval_interval == 0:
            policy.eval()
            val_rewards = []
            val_cds = []
            for v_full, v_partial in val_loader:
                v_full = v_full.squeeze(0)
                v_partial = v_partial.squeeze(0)
                cd_val, r_val, s_val = rollout_episode(
                    diffusion=diffusion,
                    policy=policy,
                    full=v_full,
                    partial=v_partial,
                    T=T,
                    rl_cfg=rl_cfg,
                    lambda_steps=args.lambda_steps,
                    device=device,
                )
                val_rewards.append(r_val)
                val_cds.append(cd_val)

            mean_val_reward = float(np.mean(val_rewards))
            mean_val_cd = float(np.mean(val_cds))

            print(
                f"[Eval][Episode {episode}] "
                f"val_reward={mean_val_reward:.4f} val_cd={mean_val_cd:.4f}"
            )

            if mean_val_reward > best_val_reward:
                best_val_reward = mean_val_reward
                ckpt_path = workdir / "checkpoints" / "best_policy.pt"
                torch.save(policy.state_dict(), ckpt_path)
                print(
                    f"[Eval] New best policy saved to {ckpt_path} "
                    f"(val_reward={best_val_reward:.4f})"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Diffusion model checkpoint path")
    parser.add_argument("--train_npz", type=str, required=True, help="train.npz")
    parser.add_argument("--val_npz", type=str, required=True, help="val.npz")
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--lambda_steps", type=float, default=0.002)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--actions", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=200)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
