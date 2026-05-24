from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DCNNBiLSTM(nn.Module):
    """PyTorch port of the DCNN-BiLSTM IDS architecture."""

    def __init__(self, num_classes: int = 34, input_features: int = 76):
        super().__init__()
        self.input_features = input_features
        self.same_padding = ((input_features - 1) // 2, input_features // 2)
        self.conv = nn.Conv1d(1, 64, kernel_size=input_features)
        self.conv_bn = nn.BatchNorm1d(64)
        self.bilstm_1 = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.reshape_bn = nn.BatchNorm1d(1)
        self.bilstm_2 = nn.LSTM(1, 128, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, num_classes),
        )

    def _as_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        elif x.ndim == 4:
            x = x.flatten(start_dim=1).unsqueeze(-1)
        if x.ndim != 3:
            raise ValueError(f"DCNNBiLSTM expects 2D/3D feature input, got shape={tuple(x.shape)}")
        if x.shape[1] == 1 and x.shape[2] == self.input_features:
            x = x.transpose(1, 2)
        if x.shape[1] != self.input_features:
            raise ValueError(f"DCNNBiLSTM expects {self.input_features} features, got shape={tuple(x.shape)}")
        return x

    @staticmethod
    def _last_bidir_hidden(lstm_out: tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        _, (hidden, _) = lstm_out
        return torch.cat((hidden[-2], hidden[-1]), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._as_sequence(x)
        x = x.transpose(1, 2)
        x = F.pad(x, self.same_padding)
        x = F.relu(self.conv(x))
        x = self.conv_bn(x)
        x = x.transpose(1, 2)

        x = self._last_bidir_hidden(self.bilstm_1(x))
        x = x.unsqueeze(-1)
        x = self.reshape_bn(x.transpose(1, 2)).transpose(1, 2)
        x = self._last_bidir_hidden(self.bilstm_2(x))
        return self.classifier(x)
