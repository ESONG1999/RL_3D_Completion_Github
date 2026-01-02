#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.utils.seed import set_seed
from src.utils.device import get_device
from src.models.diffusion_completion import DiffusionCompletionModel, DiffusionConfig


class CompletionDataset(Dataset):
    def __init__(self, npz_path: str):
        arr = np.load(npz_path)
        self.full = arr["full_points"].astype("float32")  # (N, n_full, 3)
        self.partial = arr["partial_points"].astype("float32")  # (N, n_partial, 3)
        self.labels = arr["labels"].astype("int64")

    def __len__(self):
        return self.full.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.full[idx]),
            torch.from_numpy(self.partial[idx]),
            int(self.labels[idx]),
        )


def train(args):
    set_seed(args.seed)
    device = get_device(args.device)

    train_ds = CompletionDataset(args.train_npz)
    val_ds = CompletionDataset(args.val_npz)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

    diff_cfg = DiffusionConfig(num_steps=args.num_steps)
    model = DiffusionCompletionModel(diff_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    workdir = Path(args.workdir)
    (workdir / "checkpoints").mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    log = {"train_loss": [], "val_loss": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for full, partial, _ in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            full = full.to(device)
            partial = partial.to(device)

            B = full.shape[0]
            t = torch.randint(0, args.num_steps, (B,), device=device, dtype=torch.long)

            loss = model(full, partial, t)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for full, partial, _ in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                full = full.to(device)
                partial = partial.to(device)
                B = full.shape[0]
                t = torch.randint(0, args.num_steps, (B,), device=device, dtype=torch.long)
                loss = model(full, partial, t)
                val_losses.append(loss.item())

        mean_train = float(np.mean(train_losses))
        mean_val = float(np.mean(val_losses))
        log["train_loss"].append(mean_train)
        log["val_loss"].append(mean_val)

        print(f"[Epoch {epoch}] train_loss={mean_train:.4f} val_loss={mean_val:.4f}")

        # save last
        torch.save(model.state_dict(), workdir / "checkpoints" / "last.pt")

        # save best
        if mean_val < best_val:
            best_val = mean_val
            torch.save(model.state_dict(), workdir / "checkpoints" / "best.pt")
            print(f"[INFO] New best val loss: {best_val:.4f}")

        with (workdir / "log.json").open("w") as f:
            json.dump(log, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_npz", type=str, required=True)
    parser.add_argument("--val_npz", type=str, required=True)
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
