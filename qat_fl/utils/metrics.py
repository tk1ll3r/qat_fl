from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QuantizationMetrics:
    error_norm: float
    relative_error: float
    delta_norm: float
    communicated_bits: int


def delta_error_metrics(original: dict[str, torch.Tensor], restored: dict[str, torch.Tensor], bits: int) -> QuantizationMetrics:
    error_sq = 0.0
    delta_sq = 0.0
    values = 0
    for key, tensor in original.items():
        if not tensor.is_floating_point():
            continue
        diff = restored[key].to(tensor.dtype) - tensor
        error_sq += float(torch.sum(diff * diff))
        delta_sq += float(torch.sum(tensor * tensor))
        values += tensor.numel()
    error_norm = error_sq ** 0.5
    delta_norm = delta_sq ** 0.5
    relative = error_norm / (delta_norm + 1e-12)
    return QuantizationMetrics(
        error_norm=error_norm,
        relative_error=relative,
        delta_norm=delta_norm,
        communicated_bits=values * bits + values + 32,
    )

