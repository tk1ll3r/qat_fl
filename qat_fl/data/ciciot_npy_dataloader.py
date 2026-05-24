from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class NpyClassificationDataset(Dataset):
    def __init__(self, features_path: str | Path, labels_path: str | Path):
        self.features = np.load(features_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")
        if len(self.features) != len(self.labels):
            raise ValueError(f"feature/label length mismatch: {features_path}, {labels_path}")

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(np.array(self.features[index], dtype=np.float32, copy=True)).unsqueeze(0)
        y = torch.as_tensor(int(self.labels[index]), dtype=torch.long)
        return x, y


def load_metadata(data_dir: str | Path) -> dict[str, Any]:
    metadata_path = Path(data_dir) / "metadata.json"
    return json.loads(metadata_path.read_text())


def ciciot_npy_client_loaders(
    *,
    data_dir: str | Path,
    num_clients: int,
    batch_size: int,
    seed: int,
) -> list[DataLoader]:
    root = Path(data_dir)
    loaders = []
    for cid in range(num_clients):
        dataset = NpyClassificationDataset(root / f"client_{cid}_X.npy", root / f"client_{cid}_y.npy")
        loaders.append(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed + cid),
            )
        )
    return loaders

def ciciot_npy_test_loader(
    *,
    data_dir: str | Path,
    batch_size: int,
) -> DataLoader:
    root = Path(data_dir)
    dataset = NpyClassificationDataset(root / "global_test_X.npy", root / "global_test_y.npy")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
