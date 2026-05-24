from __future__ import annotations

import torch


def aggregate_deltas(
    reference_state: dict[str, torch.Tensor],
    deltas: list[dict[str, torch.Tensor]],
    sizes: list[int],
    *,
    mode: str = "uniform",
) -> dict[str, torch.Tensor]:
    if not deltas:
        raise ValueError("at least one client delta is required")
    if mode not in {"uniform", "weighted_by_data_size"}:
        raise ValueError("mode must be 'uniform' or 'weighted_by_data_size'")
    if mode == "uniform":
        weights = [1.0 / len(deltas)] * len(deltas)
    else:
        total = float(sum(sizes))
        weights = [size / total for size in sizes]

    new_state = {}
    for key, ref in reference_state.items():
        if ref.is_floating_point():
            update = torch.zeros_like(ref)
            for delta, weight in zip(deltas, weights):
                update += delta[key].to(ref.device, dtype=ref.dtype) * weight
            new_state[key] = ref + update
        else:
            new_state[key] = ref
    return new_state

