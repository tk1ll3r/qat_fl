from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QuantizedTensor:
    values: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    num_bits: int
    shape: torch.Size
    signed: bool = True


def qrange(num_bits: int, signed: bool = True) -> tuple[int, int]:
    if num_bits < 1:
        raise ValueError("num_bits must be >= 1")
    if signed:
        return -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1
    return 0, 2**num_bits - 1


def uniform_quantize(tensor: torch.Tensor, num_bits: int, *, signed: bool = True, eps: float = 1e-12) -> QuantizedTensor:
    if not tensor.is_floating_point():
        raise TypeError("uniform_quantize expects a floating point tensor")
    qmin, qmax = qrange(num_bits, signed=signed)
    t_min = tensor.min()
    t_max = tensor.max()
    scale = (t_max - t_min).clamp_min(eps) / float(qmax - qmin)
    zero_point = torch.round(torch.tensor(qmin, device=tensor.device, dtype=tensor.dtype) - t_min / scale)
    q = torch.round(tensor / scale + zero_point).clamp(qmin, qmax)
    dtype = torch.int8 if num_bits <= 8 else torch.int16
    return QuantizedTensor(q.to(dtype).cpu(), scale.detach().cpu(), zero_point.detach().cpu(), num_bits, tensor.shape, signed)


def uniform_dequantize(qtensor: QuantizedTensor, *, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    values = qtensor.values.to(device=device, dtype=dtype)
    scale = qtensor.scale.to(device=device, dtype=dtype)
    zero_point = qtensor.zero_point.to(device=device, dtype=dtype)
    return (values - zero_point) * scale


def quantize_state_delta(
    local_state: dict[str, torch.Tensor],
    global_state: dict[str, torch.Tensor],
    num_bits: int,
) -> dict[str, QuantizedTensor | torch.Tensor]:
    quantized: dict[str, QuantizedTensor | torch.Tensor] = {}
    for key, local_tensor in local_state.items():
        global_tensor = global_state[key].to(local_tensor.device)
        if local_tensor.is_floating_point():
            quantized[key] = uniform_quantize(local_tensor - global_tensor, num_bits)
        else:
            quantized[key] = local_tensor.detach().cpu()
    return quantized


def dequantize_state_delta(
    quantized_delta: dict[str, QuantizedTensor | torch.Tensor],
    reference_state: dict[str, torch.Tensor],
    *,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    delta: dict[str, torch.Tensor] = {}
    for key, value in quantized_delta.items():
        ref = reference_state[key]
        target_device = device or ref.device
        if isinstance(value, QuantizedTensor):
            delta[key] = uniform_dequantize(value, device=target_device, dtype=ref.dtype)
        else:
            delta[key] = value.to(device=target_device)
    return delta


def full_precision_state_delta(
    local_state: dict[str, torch.Tensor],
    global_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    delta = {}
    for key, local_tensor in local_state.items():
        global_tensor = global_state[key].to(local_tensor.device)
        delta[key] = (local_tensor - global_tensor).detach().cpu() if local_tensor.is_floating_point() else local_tensor.detach().cpu()
    return delta

