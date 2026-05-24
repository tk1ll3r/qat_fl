from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset


def synthetic_image_client_loaders(
    *,
    num_clients: int,
    samples_per_client: int,
    num_classes: int,
    batch_size: int,
    seed: int,
) -> list[DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    loaders = []
    for cid in range(num_clients):
        labels = torch.randint(0, num_classes, (samples_per_client,), generator=generator)
        images = torch.randn(samples_per_client, 1, 28, 28, generator=generator)
        marker = labels.float().view(-1, 1, 1, 1) / max(num_classes - 1, 1)
        images[:, :, :4, :4] += marker
        dataset = TensorDataset(images, labels)
        loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed + cid)))
    return loaders


def synthetic_ids_client_loaders(
    *,
    num_clients: int,
    samples_per_client: int,
    num_classes: int,
    batch_size: int,
    seed: int,
    input_features: int = 76,
) -> list[DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    loaders = []
    segment = max(input_features // max(num_classes, 1), 1)
    for cid in range(num_clients):
        labels = torch.randint(0, num_classes, (samples_per_client,), generator=generator)
        features = torch.randn(samples_per_client, input_features, 1, generator=generator) * 0.2
        for class_id in range(num_classes):
            start = class_id * segment
            stop = input_features if class_id == num_classes - 1 else min(start + segment, input_features)
            features[labels == class_id, start:stop, 0] += 1.0
        dataset = TensorDataset(features, labels)
        loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed + cid)))
    return loaders


def synthetic_test_loader(*, samples: int, num_classes: int, batch_size: int, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, num_classes, (samples,), generator=generator)
    images = torch.randn(samples, 1, 28, 28, generator=generator)
    images[:, :, :4, :4] += labels.float().view(-1, 1, 1, 1) / max(num_classes - 1, 1)
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)
