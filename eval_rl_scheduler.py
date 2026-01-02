#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
eval_rl_scheduler.py

Evaluate:
1) diffusion baseline
2) RL scheduler

output:
- baseline: mean Chamfer distance, mean steps
- RL: mean Chamfer distance, mean steps
- ΔCD, Δsteps
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
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


# -----------------------------
# Dataset
# -----------------------------

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


# -----------------------------
# Sampling helpers
# -----------------------------

@torch.no_grad()
def sample_with_baseline(
    diffusion: DiffusionCompletionModel,
    full: torch.Tensor,
    partial: torch.Tensor,
    T: int,
    device: torch.device,
):
    diffusion.eval()

    full = full.unsqueeze(0).to(device)      # (1, N, 3)
    partial = partial.unsqueeze(0).to(device)  # (1, Np, 3)

    cond = diffusion.encoder(partial)

    x = torch.randn_like(full)
    steps = 0

    time_dim = diffusion.time_mlp[0].in_features

    for t in range(T - 1, -1, -1):
        t_tensor = torch.tensor([t], device=device, dtype=torch.long)

        t_emb = sinusoidal_time_embedding(t_tensor, time_dim)
        t_emb = diffusion.time_mlp(t_emb)
        eps = diffusion.denoiser(x, cond, t_emb)

        a_t = diffusion.alphas_cumprod[t_tensor].view(-1, 1, 1)

        if t > 0:
            t_prev = t - 1
            t_prev_tensor = torch.tensor([t_prev], device=device, dtype=torch.long)
            a_prev = diffusion.alphas_cumprod[t_prev_tensor].view(-1, 1, 1)
            x0_pred = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t + 1e-8)
            x = torch.sqrt(a_prev) * x0_pred + torch.sqrt(1 - a_prev) * eps
        else:
            # final x0
            x0 = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t + 1e-8)

        steps += 1

    cd = chamfer_distance(x0, full).item()
    return cd, steps


@torch.no_grad()
def sample_with_rl(
    diffusion: DiffusionCompletionModel,
    policy: RLScheduler,
    rl_cfg: RLSchedulerConfig,
    full: torch.Tensor,
    partial: torch.Tensor,
    T: int,
    device: torch.device,
    lambda_steps: float,
):
    diffusion.eval()
    policy.eval()

    full = full.unsqueeze(0).to(device)      # (1, N, 3)
    partial = partial.unsqueeze(0).to(device)  # (1, Np, 3)

    x = torch.randn_like(full)
    t = T - 1
    steps = 0

    cond = diffusion.encoder(partial)

    time_dim = diffusion.time_mlp[0].in_features

    while t > 0 and steps < rl_cfg.max_steps_per_episode:
        from train_rl_scheduler import build_state
        state = build_state(x, t, T)  # (1, state_dim)

        action_idx, _ = policy.select_action(state)
        skip = rl_cfg.actions[action_idx.item()]

        t_tensor = torch.tensor([t], device=device, dtype=torch.long)
        t_prev = max(t - skip, 0)
        t_prev_tensor = torch.tensor([t_prev], device=device, dtype=torch.long)

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
    t_emb0 = sinusoidal_time_embedding(t_tensor, time_dim)
    t_emb0 = diffusion.time_mlp(t_emb0)
    eps0 = diffusion.denoiser(x, cond, t_emb0)
    a0 = diffusion.alphas_cumprod[t_tensor].view(-1, 1, 1)
    x0 = (x - torch.sqrt(1 - a0) * eps0) / torch.sqrt(a0 + 1e-8)

    cd = chamfer_distance(x0, full).item()
    reward = -cd - lambda_steps * steps

    return cd, reward, steps


# -----------------------------
# main eval
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Diffusion Model Baseline checkpoint path")
    parser.add_argument("--policy_ckpt", type=str, default=None, help="RL scheduler checkpoint path (best_policy.pt)")
    parser.add_argument("--test_npz", type=str, required=True, help="test.npz path")
    parser.add_argument("--num_steps", type=int, default=1000, help="Diffusion steps T")
    parser.add_argument("--actions", type=int, nargs="+", default=[1, 2, 4], help="RL skip actions")
    parser.add_argument("--lambda_steps", type=float, default=0.002, help="steps penalty")
    parser.add_argument("--max_steps", type=int, default=50, help="RL max episode steps")
    parser.add_argument("--batch_size", type=int, default=1, help="DataLoader batch_size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_eval", type=int, default=None, help="Evaluate number of test samples, None for all")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    diff_cfg = DiffusionConfig(num_steps=args.num_steps)
    diffusion = DiffusionCompletionModel(diff_cfg).to(device)
    diffusion.load_state_dict(torch.load(args.ckpt, map_location=device))
    diffusion.eval()

    policy = None
    rl_cfg = None
    if args.policy_ckpt is not None:
        rl_cfg = RLSchedulerConfig(actions=args.actions)
        rl_cfg.max_steps_per_episode = args.max_steps
        policy = RLScheduler(rl_cfg).to(device)
        policy.load_state_dict(torch.load(args.policy_ckpt, map_location=device))
        policy.eval()
        print(f"[INFO] Loaded RL policy from {args.policy_ckpt}")
    else:
        print("[INFO] No policy_ckpt provided, only evaluating baseline diffusion.")

    test_ds = CompletionDataset(args.test_npz)
    if args.num_eval is not None:
        indices = np.arange(len(test_ds))[: args.num_eval]
        test_ds = torch.utils.data.Subset(test_ds, indices)
        print(f"[INFO] Test on {len(test_ds)} test samples")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    print("[INFO] Evaluating baseline diffusion...")
    base_cds = []
    base_steps = []
    for full, partial in tqdm(test_loader, desc="Baseline"):
        B = full.shape[0]
        for i in range(B):
            cd, steps = sample_with_baseline(
                diffusion=diffusion,
                full=full[i],
                partial=partial[i],
                T=args.num_steps,
                device=device,
            )
            base_cds.append(cd)
            base_steps.append(steps)

    base_cd_mean = float(np.mean(base_cds))
    base_cd_std = float(np.std(base_cds))
    base_steps_mean = float(np.mean(base_steps))

    print(
        f"[RESULT][Baseline] "
        f"CD: {base_cd_mean:.6f} ± {base_cd_std:.6f}, "
        f"Steps: {base_steps_mean:.2f}"
    )

    if policy is not None:
        print("[INFO] Evaluating RL scheduler...")
        rl_cds = []
        rl_steps = []
        rl_rewards = []

        for full, partial in tqdm(test_loader, desc="RL"):
            B = full.shape[0]
            for i in range(B):
                cd, reward, steps = sample_with_rl(
                    diffusion=diffusion,
                    policy=policy,
                    rl_cfg=rl_cfg,
                    full=full[i],
                    partial=partial[i],
                    T=args.num_steps,
                    device=device,
                    lambda_steps=args.lambda_steps,
                )
                rl_cds.append(cd)
                rl_steps.append(steps)
                rl_rewards.append(reward)

        rl_cd_mean = float(np.mean(rl_cds))
        rl_cd_std = float(np.std(rl_cds))
        rl_steps_mean = float(np.mean(rl_steps))
        rl_reward_mean = float(np.mean(rl_rewards))

        print(
            f"[RESULT][RL] "
            f"CD: {rl_cd_mean:.6f} ± {rl_cd_std:.6f}, "
            f"Steps: {rl_steps_mean:.2f}, "
            f"Reward: {rl_reward_mean:.6f}"
        )

        delta_cd = rl_cd_mean - base_cd_mean
        delta_steps = rl_steps_mean - base_steps_mean
        print(
            f"[RESULT][Delta RL - Baseline] "
            f"ΔCD={delta_cd:.6f} (Negative value means RL is better), "
            f"ΔSteps={delta_steps:.2f} (Negative value means RL save more steps)"
        )


if __name__ == "__main__":
    main()
