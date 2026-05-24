from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CICIoT client CSV files to scaled mmap-friendly .npy arrays.")
    parser.add_argument("--input-dir", required=True, help="Directory with client_0.csv ... client_N.csv and global_test.csv.")
    parser.add_argument("--output-dir", required=True, help="Directory where .npy arrays and metadata.json will be written.")
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--test-file", default="global_test.csv")
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--max-class-weight", type=float, default=20.0)
    return parser.parse_args()


def csv_files(input_dir: Path, num_clients: int, test_file: str) -> tuple[list[tuple[str, Path]], tuple[str, Path]]:
    client_files = [(f"client_{cid}", input_dir / f"client_{cid}.csv") for cid in range(num_clients)]
    test = ("global_test", input_dir / test_file)
    for _, path in [*client_files, test]:
        if not path.exists():
            raise FileNotFoundError(path)
    return client_files, test


def scan_client_csvs(
    files: list[tuple[str, Path]],
    label_column: str,
    chunksize: int,
    class_weight_power: float,
    max_class_weight: float | None,
):
    labels: set[str] = set()
    rows: dict[str, int] = {}
    feature_columns: list[str] | None = None
    label_counts: dict[str, int] = {}
    scaler = StandardScaler()

    for name, path in files:
        total_rows = 0
        for chunk in pd.read_csv(path, chunksize=chunksize):
            if label_column not in chunk.columns:
                raise ValueError(f"{path} missing label column {label_column!r}")
            current_features = [col for col in chunk.columns if col != label_column]
            if feature_columns is None:
                feature_columns = current_features
            elif current_features != feature_columns:
                raise ValueError(f"{path} feature columns do not match first CSV")

            value_counts = chunk[label_column].astype(str).value_counts()
            labels.update(value_counts.index)
            for label, count in value_counts.items():
                label_counts[label] = label_counts.get(label, 0) + int(count)
            scaler.partial_fit(chunk[feature_columns].to_numpy())
            total_rows += len(chunk)
        rows[name] = total_rows

    if feature_columns is None:
        raise ValueError("no rows found")

    sorted_labels = sorted(labels)
    label_map = {label: idx for idx, label in enumerate(sorted_labels)}
    total = sum(label_counts.values())
    num_classes = len(sorted_labels)
    class_weights = np.asarray(
        [(total / (num_classes * label_counts[label])) ** class_weight_power for label in sorted_labels],
        dtype=np.float32,
    )
    class_weights = class_weights / class_weights.mean()
    if max_class_weight is not None:
        class_weights = np.clip(class_weights, None, max_class_weight)

    return feature_columns, label_map, rows, label_counts, class_weights, scaler


def count_rows(path: Path, chunksize: int) -> int:
    return sum(len(chunk) for chunk in pd.read_csv(path, chunksize=chunksize))


def convert_one(
    *,
    name: str,
    path: Path,
    output_dir: Path,
    feature_columns: list[str],
    label_column: str,
    label_map: dict[str, int],
    rows: int,
    chunksize: int,
    scaler: StandardScaler,
) -> None:
    x_path = output_dir / f"{name}_X.npy"
    y_path = output_dir / f"{name}_y.npy"
    x_out = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32, shape=(rows, len(feature_columns)))
    y_out = np.lib.format.open_memmap(y_path, mode="w+", dtype=np.int64, shape=(rows,))
    offset = 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        stop = offset + len(chunk)
        x_out[offset:stop] = scaler.transform(chunk[feature_columns].to_numpy()).astype(np.float32, copy=False)
        labels = chunk[label_column].astype(str).map(label_map)
        if labels.isna().any():
            bad = sorted(chunk.loc[labels.isna(), label_column].astype(str).unique())
            raise ValueError(f"{path} has unknown labels: {bad[:10]}")
        y_out[offset:stop] = labels.to_numpy(dtype=np.int64, copy=False)
        offset = stop
    x_out.flush()
    y_out.flush()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    client_files, test_file = csv_files(input_dir, args.num_clients, args.test_file)
    max_class_weight = None if args.max_class_weight <= 0 else args.max_class_weight
    feature_columns, label_map, rows, label_counts, class_weights, scaler = scan_client_csvs(
        client_files,
        args.label_column,
        args.chunksize,
        args.class_weight_power,
        max_class_weight,
    )
    rows[test_file[0]] = count_rows(test_file[1], args.chunksize)

    metadata = {
        "format": "ciciot_npy_v1",
        "source": "aligned_with_feder_learn_data_loader",
        "num_clients": args.num_clients,
        "num_features": len(feature_columns),
        "num_classes": len(label_map),
        "feature_columns": feature_columns,
        "label_column": args.label_column,
        "label_map": label_map,
        "label_counts": label_counts,
        "class_weights": class_weights.tolist(),
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "rows": rows,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    for name, path in [*client_files, test_file]:
        print(f"converting {path} -> {output_dir / name}_X.npy / {output_dir / name}_y.npy")
        convert_one(
            name=name,
            path=path,
            output_dir=output_dir,
            feature_columns=feature_columns,
            label_column=args.label_column,
            label_map=label_map,
            rows=rows[name],
            chunksize=args.chunksize,
            scaler=scaler,
        )


if __name__ == "__main__":
    main()
