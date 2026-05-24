from __future__ import annotations

from dataclasses import dataclass

import torch

from qat_fl.fl.client import FLClient
from qat_fl.fl.server import FLServer


@dataclass
class RoundResult:
    round: int
    train_loss: float
    num_bits: int
    selected_clients: list[int]
    communication_bits: int
    cumulative_communication_bits: int
    relative_quantization_error: float


class FederatedTrainer:
    def __init__(self, server: FLServer, client_loaders: list, device: torch.device, config: dict):
        self.server = server
        self.clients = [FLClient(cid, loader, device) for cid, loader in enumerate(client_loaders)]
        self.device = device
        self.config = config
        self.cumulative_bits = 0

    def run_round(self, round_id: int) -> RoundResult:
        bits = int(self.config["num_bits"])
        selected = self.server.select_clients(self.config["num_clients"], self.config["clients_per_round"])
        updates = [
            self.clients[cid].train_update(
                self.server.model,
                lr=float(self.config["lr"]),
                tau_epochs=int(self.config["tau_epochs"]),
                qat_epochs=int(self.config["qat_epochs"]),
                num_bits=bits,
                optimizer_name=str(self.config.get("optimizer", "sgd")),
            )
            for cid in selected
        ]
        self.server.apply_client_updates(updates, self.device)
        train_loss = sum(update.train_loss * update.num_samples for update in updates) / max(sum(update.num_samples for update in updates), 1)
        bits_this_round = sum(update.metrics.communicated_bits for update in updates if update.metrics is not None)
        rel_errors = [update.metrics.relative_error for update in updates if update.metrics is not None]
        self.cumulative_bits += bits_this_round
        return RoundResult(
            round=round_id,
            train_loss=train_loss,
            num_bits=bits,
            selected_clients=selected,
            communication_bits=bits_this_round,
            cumulative_communication_bits=self.cumulative_bits,
            relative_quantization_error=sum(rel_errors) / len(rel_errors) if rel_errors else 0.0,
        )
