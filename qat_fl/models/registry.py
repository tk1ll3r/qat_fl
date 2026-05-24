from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from qat_fl.models.cnn_mnist import TinyCNN
from qat_fl.models.dcnn_bilstm import DCNNBiLSTM


@dataclass(frozen=True)
class ModelSpec:
    name: str
    builder: Callable[..., object]
    paper_note: str


_REGISTRY: dict[str, ModelSpec] = {}


def register_model(spec: ModelSpec) -> None:
    key = spec.name.lower()
    if key in _REGISTRY:
        raise ValueError(f"model already registered: {key}")
    _REGISTRY[key] = spec


def model_names() -> list[str]:
    return sorted(_REGISTRY)


def create_model(name: str, **kwargs):
    try:
        return _REGISTRY[name.lower()].builder(**kwargs)
    except KeyError as exc:
        raise ValueError(f"unknown model={name!r}; available={model_names()}") from exc


register_model(ModelSpec("tiny_cnn", TinyCNN, "Small CNN smoke model for MNIST-style experiments."))
register_model(
    ModelSpec(
        "dcnn_bilstm",
        DCNNBiLSTM,
        "PyTorch port of DCNNBiLSTM: Conv1D + stacked BiLSTM + dense IDS classifier.",
    )
)
