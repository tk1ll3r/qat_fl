from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn

from qat_fl.quantization.fake_quant import prepare_weight_qat, set_fake_quant
from qat_fl.quantization.uniform import QuantizedTensor, full_precision_state_delta, quantize_state_delta
from qat_fl.utils.metrics import QuantizationMetrics, delta_error_metrics


@dataclass
class ClientUpdate:
    payload: dict
    num_samples: int
    train_loss: float
    metrics: QuantizationMetrics | None


def _run_epochs(model: nn.Module, loader, optimizer, criterion, device: torch.device, epochs: int) -> tuple[float, int]:
    total_loss = 0.0
    total_seen = 0
    model.train()
    for _ in range(max(epochs, 0)):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * y.numel()
            total_seen += y.numel()
    return total_loss / max(total_seen, 1), total_seen


def _loader_num_samples(loader, fallback: int) -> int:
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return fallback
    try:
        return int(len(dataset))
    except TypeError:
        return fallback


def _make_optimizer(model: nn.Module, name: str, lr: float) -> torch.optim.Optimizer:
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    raise ValueError("optimizer must be sgd or adam")


class FLClient:
    def __init__(self, client_id: int, loader, device: torch.device):
        self.client_id = client_id
        self.loader = loader
        self.device = device

    def train_update(
        self,
        global_model: nn.Module,
        *,
        lr: float,
        tau_epochs: int,
        qat_epochs: int,
        num_bits: int,
        optimizer_name: str = "sgd",
    ) -> ClientUpdate:
        model = copy.deepcopy(global_model).to(self.device)
        global_state = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}
        criterion = nn.CrossEntropyLoss()
        optimizer = _make_optimizer(model, optimizer_name, lr)

        tau_loss, tau_seen = _run_epochs(model, self.loader, optimizer, criterion, self.device, tau_epochs)
        prepare_weight_qat(model, num_bits)
        set_fake_quant(model, True)
        optimizer = _make_optimizer(model, optimizer_name, lr)
        qat_loss, qat_seen = _run_epochs(model, self.loader, optimizer, criterion, self.device, qat_epochs)
        set_fake_quant(model, False)
        seen = tau_seen + qat_seen
        avg_loss = ((tau_loss * tau_seen) + (qat_loss * qat_seen)) / max(seen, 1)
        num_samples = _loader_num_samples(self.loader, seen)

        local_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        payload = quantize_state_delta(local_state, global_state, num_bits)
        restored = {}
        for key, value in payload.items():
            if isinstance(value, QuantizedTensor):
                from qat_fl.quantization.uniform import uniform_dequantize

                restored[key] = uniform_dequantize(value, dtype=global_state[key].dtype)
            else:
                restored[key] = value
        original_delta = full_precision_state_delta(local_state, global_state)
        metrics = delta_error_metrics(original_delta, restored, num_bits)
        return ClientUpdate(payload, num_samples, avg_loss, metrics)
