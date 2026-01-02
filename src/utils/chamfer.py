# src/utils/chamfer.py
import torch
from torch import Tensor


@torch.no_grad()
def chamfer_distance(x: Tensor, y: Tensor) -> Tensor:
    """
    x: (B, N, 3)
    y: (B, M, 3)
    Returns mean Chamfer distance over batch.
    """
    B, N, _ = x.shape
    _, M, _ = y.shape

    # (B, N, M)
    x_exp = x.unsqueeze(2)  # (B, N, 1, 3)
    y_exp = y.unsqueeze(1)  # (B, 1, M, 3)
    dist = torch.sum((x_exp - y_exp) ** 2, dim=-1)

    cd_xy = dist.min(dim=2).values.mean(dim=1)  # (B,)
    cd_yx = dist.min(dim=1).values.mean(dim=1)
    cd = cd_xy + cd_yx
    return cd.mean()
