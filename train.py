from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from qat_fl.data.ciciot_npy_dataloader import ciciot_npy_client_loaders, load_metadata
from qat_fl.data.partition import synthetic_ids_client_loaders, synthetic_image_client_loaders
from qat_fl.fl.server import FLServer
from qat_fl.fl.trainer import FederatedTrainer
from qat_fl.models.registry import create_model
from qat_fl.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="QAT-FL smoke trainer")
    parser.add_argument("--config", default="configs/synthetic_qat.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = config.get("dataset", "synthetic_image")

    if dataset == "synthetic_ids":
        loader_kwargs = {
            "num_clients": int(config["num_clients"]),
            "samples_per_client": int(config["samples_per_client"]),
            "num_classes": int(config["num_classes"]),
            "batch_size": int(config["batch_size"]),
            "seed": int(config["seed"]),
        }
        input_features = int(config.get("input_features", 76))
        loaders = synthetic_ids_client_loaders(**loader_kwargs, input_features=input_features)
        model_kwargs = {"num_classes": int(config["num_classes"]), "input_features": input_features}
    elif dataset == "ciciot_npy":
        metadata = load_metadata(config["data_dir"])
        config["num_clients"] = int(metadata["num_clients"])
        config["num_classes"] = int(metadata["num_classes"])
        model_kwargs = {"num_classes": int(metadata["num_classes"]), "input_features": int(metadata["num_features"])}
        loaders = ciciot_npy_client_loaders(
            data_dir=config["data_dir"],
            num_clients=int(config["num_clients"]),
            batch_size=int(config["batch_size"]),
            seed=int(config["seed"]),
        )
    elif dataset == "synthetic_image":
        loader_kwargs = {
            "num_clients": int(config["num_clients"]),
            "samples_per_client": int(config["samples_per_client"]),
            "num_classes": int(config["num_classes"]),
            "batch_size": int(config["batch_size"]),
            "seed": int(config["seed"]),
        }
        loaders = synthetic_image_client_loaders(**loader_kwargs)
        model_kwargs = {"num_classes": int(config["num_classes"])}
    else:
        raise ValueError("dataset must be synthetic_image, synthetic_ids, or ciciot_npy")

    model = create_model(config["model"], **model_kwargs).to(device)
    server = FLServer(model, aggregation=config["aggregation"], seed=int(config["seed"]))
    trainer = FederatedTrainer(server, loaders, device, config)
    previous_loss = None
    for round_id in range(1, int(config["rounds"]) + 1):
        result = trainer.run_round(round_id, previous_loss)
        previous_loss = result.train_loss
        print(
            f"round={result.round} loss={result.train_loss:.4f} bits={result.num_bits} "
            f"comm_bits={result.communication_bits} rel_qerr={result.relative_quantization_error:.4f} "
            f"clients={result.selected_clients}"
        )


if __name__ == "__main__":
    main()
