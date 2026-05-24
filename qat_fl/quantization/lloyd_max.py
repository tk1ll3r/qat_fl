from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LloydMaxCodebook:
    levels: torch.Tensor
    thresholds: torch.Tensor


def fit_lloyd_max(tensor: torch.Tensor, num_levels: int, iterations: int = 20) -> LloydMaxCodebook:
    flat = tensor.detach().flatten().float()
    if num_levels < 2:
        raise ValueError("num_levels must be >= 2")
    levels = torch.linspace(flat.min(), flat.max(), num_levels, device=flat.device)
    for _ in range(iterations):
        distances = torch.abs(flat[:, None] - levels[None, :])
        assign = torch.argmin(distances, dim=1)
        new_levels = levels.clone()
        for idx in range(num_levels):
            bucket = flat[assign == idx]
            if bucket.numel() > 0:
                new_levels[idx] = bucket.mean()
        if torch.allclose(levels, new_levels):
            break
        levels = new_levels
    thresholds = (levels[:-1] + levels[1:]) / 2
    return LloydMaxCodebook(levels=levels.cpu(), thresholds=thresholds.cpu())


def lloyd_max_dequantize(indices: torch.Tensor, codebook: LloydMaxCodebook) -> torch.Tensor:
    return codebook.levels.to(indices.device)[indices.long()]

