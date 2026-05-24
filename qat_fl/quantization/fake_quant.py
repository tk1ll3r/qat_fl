from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from qat_fl.quantization.uniform import qrange


def fake_quantize_ste(tensor: torch.Tensor, num_bits: int, eps: float = 1e-12) -> torch.Tensor:
    qmin, qmax = qrange(num_bits, signed=True)
    t_min = tensor.min()
    t_max = tensor.max()
    scale = (t_max - t_min).clamp_min(eps) / float(qmax - qmin)
    zero_point = torch.round(torch.tensor(qmin, device=tensor.device, dtype=tensor.dtype) - t_min / scale)
    q = torch.round(tensor / scale + zero_point).clamp(qmin, qmax)
    dequant = (q - zero_point) * scale
    return tensor + (dequant - tensor).detach()


class FakeQuantLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, num_bits: int = 4):
        super().__init__(in_features, out_features, bias=bias)
        self.num_bits = num_bits
        self.fake_quant_enabled = True

    @classmethod
    def from_float(cls, module: nn.Linear, num_bits: int) -> "FakeQuantLinear":
        wrapped = cls(module.in_features, module.out_features, module.bias is not None, num_bits)
        wrapped.to(device=module.weight.device, dtype=module.weight.dtype)
        with torch.no_grad():
            wrapped.weight.copy_(module.weight)
            if module.bias is not None:
                wrapped.bias.copy_(module.bias)
        return wrapped

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = fake_quantize_ste(self.weight, self.num_bits) if self.fake_quant_enabled else self.weight
        return F.linear(x, weight, self.bias)


class FakeQuantConv2d(nn.Conv2d):
    def __init__(self, *args, num_bits: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_bits = num_bits
        self.fake_quant_enabled = True

    @classmethod
    def from_float(cls, module: nn.Conv2d, num_bits: int) -> "FakeQuantConv2d":
        wrapped = cls(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            num_bits=num_bits,
        )
        wrapped.to(device=module.weight.device, dtype=module.weight.dtype)
        with torch.no_grad():
            wrapped.weight.copy_(module.weight)
            if module.bias is not None:
                wrapped.bias.copy_(module.bias)
        return wrapped

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = fake_quantize_ste(self.weight, self.num_bits) if self.fake_quant_enabled else self.weight
        return self._conv_forward(x, weight, self.bias)


class FakeQuantConv1d(nn.Conv1d):
    def __init__(self, *args, num_bits: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_bits = num_bits
        self.fake_quant_enabled = True

    @classmethod
    def from_float(cls, module: nn.Conv1d, num_bits: int) -> "FakeQuantConv1d":
        wrapped = cls(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            num_bits=num_bits,
        )
        wrapped.to(device=module.weight.device, dtype=module.weight.dtype)
        with torch.no_grad():
            wrapped.weight.copy_(module.weight)
            if module.bias is not None:
                wrapped.bias.copy_(module.bias)
        return wrapped

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = fake_quantize_ste(self.weight, self.num_bits) if self.fake_quant_enabled else self.weight
        return self._conv_forward(x, weight, self.bias)


def prepare_weight_qat(model: nn.Module, num_bits: int) -> int:
    replaced = 0
    for name, child in list(model.named_children()):
        if isinstance(child, (FakeQuantLinear, FakeQuantConv1d, FakeQuantConv2d)):
            child.num_bits = num_bits
            continue
        if isinstance(child, nn.Linear):
            setattr(model, name, FakeQuantLinear.from_float(child, num_bits))
            replaced += 1
        elif isinstance(child, nn.Conv1d):
            setattr(model, name, FakeQuantConv1d.from_float(child, num_bits))
            replaced += 1
        elif isinstance(child, nn.Conv2d):
            setattr(model, name, FakeQuantConv2d.from_float(child, num_bits))
            replaced += 1
        else:
            replaced += prepare_weight_qat(child, num_bits)
    return replaced


def set_fake_quant(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, (FakeQuantLinear, FakeQuantConv1d, FakeQuantConv2d)):
            module.fake_quant_enabled = enabled
