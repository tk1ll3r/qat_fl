from __future__ import annotations

import random

import torch
import torch.nn as nn

from qat_fl.fl.fedavg import aggregate_deltas
from qat_fl.quantization.uniform import QuantizedTensor, dequantize_state_delta
from qat_fl.utils.metrics import ClassificationMetrics, evaluate_classification


class FLServer:
    def __init__(self, model: nn.Module, *, aggregation: str = "uniform", seed: int = 42):
        self.model = model
        self.aggregation = aggregation
        self.rng = random.Random(seed)

    def select_clients(self, num_clients: int, clients_per_round: int) -> list[int]:
        if clients_per_round <= 0 or clients_per_round >= num_clients:
            return list(range(num_clients))
        return sorted(self.rng.sample(range(num_clients), clients_per_round))

    def apply_client_updates(self, updates: list, device: torch.device) -> None:
        reference_state = self.model.state_dict()
        deltas = []
        sizes = []
        for update in updates:
            if any(isinstance(value, QuantizedTensor) for value in update.payload.values()):
                deltas.append(dequantize_state_delta(update.payload, reference_state, device=device))
            else:
                deltas.append({key: value.to(device=device) for key, value in update.payload.items()})
            sizes.append(update.num_samples)
        new_state = aggregate_deltas(reference_state, deltas, sizes, mode=self.aggregation)
        self.model.load_state_dict(new_state)

    def evaluate_global_model(self, loader, device: torch.device, num_classes: int) -> ClassificationMetrics:
        return evaluate_classification(self.model, loader, device, num_classes)
