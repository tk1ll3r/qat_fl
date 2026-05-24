from __future__ import annotations

import torch


def ste_round(x: torch.Tensor) -> torch.Tensor:
    rounded = torch.round(x)
    return x + (rounded - x).detach()

