from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from qat_fl.data.ciciot_csv_dataloader import CICIoTCSVDataModule
from qat_fl.data.partition import synthetic_ids_client_loaders
from qat_fl.fl.server import FLServer
from qat_fl.fl.trainer import FederatedTrainer
from qat_fl.models.registry import create_model
from qat_fl.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="QAT-FL smoke trainer")
    parser.add_argument("--config", default="configs/synthetic_qat.yaml")
    parser.add_argument("--data-dir", default=None, help="Override data_dir from the config file. Use 'auto' on Kaggle.")
    return parser.parse_args()


def _resolve_config_path(config_path: str) -> Path:
    path = Path(config_path)
    if path.exists():
        return path
    script_relative = Path(__file__).resolve().parent / config_path
    if script_relative.exists():
        return script_relative
    raise FileNotFoundError(f"Config file not found: {config_path}")


def _find_kaggle_data_dir(num_clients: int, test_file: str) -> Path:
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        raise FileNotFoundError("/kaggle/input not found. Pass --data-dir explicitly.")

    matches = []
    for client_zero in input_root.rglob("client_0.csv"):
        data_dir = client_zero.parent
        if not (data_dir / test_file).exists():
            continue
        if all((data_dir / f"client_{client_id}.csv").exists() for client_id in range(num_clients)):
            matches.append(data_dir)

    if not matches:
        raise FileNotFoundError(
            f"Could not find CICIoT CSV data under /kaggle/input with client_0.csv..client_{num_clients - 1}.csv "
            f"and {test_file}."
        )
    if len(matches) > 1:
        shown = ", ".join(str(path) for path in matches[:5])
        raise ValueError(f"Multiple Kaggle data dirs matched; pass --data-dir explicitly. Matches: {shown}")
    return matches[0]


def _resolve_data_dir(config: dict) -> str:
    data_dir = str(config["data_dir"])
    if data_dir.lower() != "auto":
        return data_dir
    return str(_find_kaggle_data_dir(int(config["num_clients"]), str(config.get("test_file", "global_test.csv"))))


def main() -> None:
    args = parse_args()
    config_path = _resolve_config_path(args.config)
    config = yaml.safe_load(config_path.read_text())
    if args.data_dir is not None:
        config["data_dir"] = args.data_dir
    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = config.get("dataset", "synthetic_ids")
    test_loader = None

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
    elif dataset == "ciciot_csv":
        config["data_dir"] = _resolve_data_dir(config)
        print(f"data_dir={config['data_dir']}")
        data_module = CICIoTCSVDataModule(
            data_dir=config["data_dir"],
            num_clients=int(config["num_clients"]),
            batch_size=int(config["batch_size"]),
            label_column=str(config.get("label_column", "label")),
            test_file=str(config.get("test_file", "global_test.csv")),
            seed=int(config["seed"]),
        )
        config["num_clients"] = data_module.num_clients
        config["num_classes"] = data_module.num_classes
        model_kwargs = {"num_classes": data_module.num_classes, "input_features": data_module.num_features}
        loaders = data_module.client_loaders()
        test_loader = data_module.test_loader()
    else:
        raise ValueError("dataset must be synthetic_ids or ciciot_csv")

    model = create_model(config["model"], **model_kwargs).to(device)
    server = FLServer(model, seed=int(config["seed"]))
    trainer = FederatedTrainer(server, loaders, device, config)
    for round_id in range(1, int(config["rounds"]) + 1):
        result = trainer.run_round(round_id)
        log = (
            f"round={result.round} train_loss={result.train_loss:.4f} bits={result.num_bits} "
            f"comm_bits={result.communication_bits} rel_qerr={result.relative_quantization_error:.4f} "
            f"clients={result.selected_clients}"
        )
        if test_loader is not None:
            metrics = server.evaluate_global_model(test_loader, device, int(config["num_classes"]))
            log += (
                f" server_test_loss={metrics.loss:.4f} server_acc={metrics.accuracy:.4f} "
                f"server_macro_p={metrics.macro_precision:.4f} server_macro_r={metrics.macro_recall:.4f} "
                f"server_macro_f1={metrics.macro_f1:.4f} server_weighted_f1={metrics.weighted_f1:.4f}"
            )
        print(log)


if __name__ == "__main__":
    main()
