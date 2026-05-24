from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class QuantizationMetrics:
    error_norm: float
    relative_error: float
    delta_norm: float
    communicated_bits: int


@dataclass(frozen=True)
class ClassificationMetrics:
    loss: float
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float


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


@torch.no_grad()
def evaluate_classification(
    model: nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
) -> ClassificationMetrics:
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_seen = 0
    correct = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

    model.eval()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        total_loss += float(criterion(logits, y))
        total_seen += y.numel()
        correct += int((pred == y).sum())
        encoded = y.cpu() * num_classes + pred.cpu()
        confusion += torch.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

    tp = confusion.diag().to(torch.float64)
    predicted_per_class = confusion.sum(dim=0).to(torch.float64)
    actual_per_class = confusion.sum(dim=1).to(torch.float64)
    precision = tp / predicted_per_class.clamp_min(1)
    recall = tp / actual_per_class.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    present = actual_per_class > 0
    weights = actual_per_class / actual_per_class.sum().clamp_min(1)

    return ClassificationMetrics(
        loss=total_loss / max(total_seen, 1),
        accuracy=correct / max(total_seen, 1),
        macro_precision=float(precision[present].mean()) if bool(present.any()) else 0.0,
        macro_recall=float(recall[present].mean()) if bool(present.any()) else 0.0,
        macro_f1=float(f1[present].mean()) if bool(present.any()) else 0.0,
        weighted_f1=float((f1 * weights).sum()),
    )
