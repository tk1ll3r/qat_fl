from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class CICIoTCSVDataset(Dataset):
    """In-memory CICIoT tensor dataset."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        label_map: dict[str, int],
        scaler: StandardScaler,
        feature_columns: list[str],
        label_column: str,
        source: Path,
    ):
        _validate_columns(source, frame, feature_columns, label_column)
        labels = _map_labels(frame[label_column], label_map, source)
        x_scaled = scaler.transform(frame[feature_columns].to_numpy(dtype=np.float32, copy=False))
        self.x_tensor = torch.as_tensor(x_scaled, dtype=torch.float32).unsqueeze(1)
        self.y_tensor = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.y_tensor.numel())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x_tensor[index], self.y_tensor[index]


class CICIoTCSVDataModule:
    """Load all CICIoT client CSVs into RAM and build FL dataloaders."""

    def __init__(
        self,
        *,
        data_dir: str | Path,
        num_clients: int,
        batch_size: int,
        label_column: str = "label",
        test_file: str = "global_test.csv",
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.num_clients = int(num_clients)
        self.batch_size = int(batch_size)
        self.label_column = label_column
        self.test_file = test_file
        self.seed = int(seed)
        self.client_num_samples: dict[int, int] = {}
        self._client_frames: dict[int, pd.DataFrame] = {}
        self._client_datasets: dict[int, CICIoTCSVDataset] = {}
        self._test_dataset: CICIoTCSVDataset | None = None

        if self.num_clients <= 0:
            raise ValueError("num_clients must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self.data_dir.exists():
            raise FileNotFoundError(f"CSV data_dir not found: {self.data_dir}")

        self.feature_columns, self.label_map, self.scaler = self._load_clients()
        self.num_features = len(self.feature_columns)
        self.num_classes = len(self.label_map)
        self._build_client_datasets()

    def metadata(self) -> dict[str, Any]:
        return {
            "num_clients": self.num_clients,
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "label_map": self.label_map,
            "client_num_samples": self.client_num_samples,
            "feature_columns": self.feature_columns,
        }

    def client_loaders(self) -> list[DataLoader]:
        return [self.get_client_loader(client_id) for client_id in range(self.num_clients)]

    def get_client_loader(self, client_id: int) -> DataLoader:
        dataset = self._client_datasets[client_id]
        generator = torch.Generator().manual_seed(self.seed + client_id)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, generator=generator)

    def test_loader(self) -> DataLoader:
        test_path = Path(self.test_file)
        if not test_path.is_absolute():
            test_path = self.data_dir / self.test_file
        if self._test_dataset is None:
            if not test_path.exists():
                raise FileNotFoundError(test_path)
            self._test_dataset = CICIoTCSVDataset(
                pd.read_csv(test_path),
                label_map=self.label_map,
                scaler=self.scaler,
                feature_columns=self.feature_columns,
                label_column=self.label_column,
                source=test_path,
            )
        return DataLoader(self._test_dataset, batch_size=self.batch_size, shuffle=False)

    def _client_path(self, client_id: int) -> Path:
        return self.data_dir / f"client_{client_id}.csv"

    def _load_clients(self) -> tuple[list[str], dict[str, int], StandardScaler]:
        label_values: set[str] = set()
        feature_columns: list[str] | None = None
        scaler = StandardScaler()

        for client_id in range(self.num_clients):
            file_path = self._client_path(client_id)
            if not file_path.exists():
                raise FileNotFoundError(f"Missing CICIoT client CSV: {file_path}")

            frame = pd.read_csv(file_path)
            if self.label_column not in frame.columns:
                raise ValueError(f"{file_path} missing label column {self.label_column!r}")

            current_features = [column for column in frame.columns if column != self.label_column]
            if feature_columns is None:
                feature_columns = current_features
            elif current_features != feature_columns:
                raise ValueError(f"{file_path} feature columns do not match the first client CSV")

            label_values.update(frame[self.label_column].astype(str).unique())
            scaler.partial_fit(frame[current_features].to_numpy(dtype=np.float32, copy=False))
            self.client_num_samples[client_id] = int(len(frame))
            self._client_frames[client_id] = frame

        if feature_columns is None:
            raise ValueError(f"No client CSVs found in {self.data_dir}")

        label_map = {label: index for index, label in enumerate(sorted(label_values))}
        return feature_columns, label_map, scaler

    def _build_client_datasets(self) -> None:
        for client_id, frame in self._client_frames.items():
            self._client_datasets[client_id] = CICIoTCSVDataset(
                frame,
                label_map=self.label_map,
                scaler=self.scaler,
                feature_columns=self.feature_columns,
                label_column=self.label_column,
                source=self._client_path(client_id),
            )
        self._client_frames.clear()


def _map_labels(labels: pd.Series, label_map: dict[str, int], file_path: Path) -> np.ndarray:
    mapped = labels.astype(str).map(label_map)
    if mapped.isna().any():
        bad = sorted(labels.loc[mapped.isna()].astype(str).unique())
        raise ValueError(f"Unknown labels in {file_path}: {bad[:10]}")
    return mapped.to_numpy(dtype=np.int64, copy=True)


def _validate_columns(file_path: Path, df: pd.DataFrame, feature_columns: list[str], label_column: str) -> None:
    if label_column not in df.columns:
        raise ValueError(f"{file_path} missing label column {label_column!r}")
    current_features = [column for column in df.columns if column != label_column]
    if current_features != feature_columns:
        raise ValueError(f"{file_path} columns do not match expected CICIoT schema")
