# QAT-FL TMLCN 2025 Rebuild

This repository is a clean rebuild scaffold for:

**Communication Efficient Federated Learning With Quantization-Aware Training Design**  
DOI: `10.1109/TMLCN.2025.3635050`

The implementation is kept narrow on purpose: FedAvg aggregation, QAT local
training, uniform delta quantization, and CSV dataloaders for CICIoT-style
experiments.

## Implemented Baseline Surface

- FedAvg aggregation weighted by client sample count.
- QAT-FL local flow: `tau` normal local epochs, then `M` fake-quantized QAT epochs.
- Uniform symmetric per-tensor quantization with communication-bit accounting.
- Synthetic smoke dataset so the repo can run without downloads.
- DCNN-BiLSTM IDS model option ported from the public Keras notebook for
  "DCNNBiLSTM: An Efficient Hybrid Deep Learning-Based Intrusion Detection System".

## Layout

```text
qat_fl/
  data/           dataset partition helpers
  fl/             client, server, FedAvg, trainer
  models/         model registry and CNNs
  quantization/   uniform delta quantization and fake quantization
  utils/          seed, metrics, checkpoint helpers
configs/          experiment config examples
tests/            focused unit tests
train.py          runnable smoke training entrypoint
```

## Quick Start

```bash
pip install -r requirements.txt
python -m pytest -q
python train.py --config configs/synthetic_qat.yaml
```

The default smoke config now uses `model: dcnn_bilstm` with synthetic IDS-like
feature vectors shaped `(76, 1)` and `num_classes: 34` for CICIoT2023-style
multi-class classification.

## DCNN-BiLSTM Port Notes

The referenced notebook architecture is mapped to PyTorch as:

`Conv1D(64, kernel_size=76, same)` -> BatchNorm -> BiLSTM(64) ->
reshape `(128, 1)` -> BatchNorm -> BiLSTM(128) -> Dense `64/32/16` with
dropout `0.1` -> `num_classes` logits.

Intentional differences from the notebook:

- PyTorch returns logits and uses `CrossEntropyLoss`; the notebook adds
  `softmax` and uses `categorical_crossentropy`. This is the numerically stable
  PyTorch convention and is equivalent for training.
- The repo keeps federated QAT training instead of the notebook's centralized
  Keras training loop, because this project is a QAT-FL scaffold.
- The referenced notebook trains a 3-class CIC-IDS-2018 DDoS subset. For
  CICIoT2023, this repo sets the output layer from `num_classes`; the default
  smoke config uses 34 classes to match the local CICIoT2023 baseline.
- The default config uses synthetic 76-feature IDS-like data, not CICIoT2023 CSV
  preprocessing. This keeps smoke tests runnable without downloading large
  research data.
- Fake-QAT now wraps `Conv1d` and dense layers. LSTM weights are still quantized
  in communicated deltas, but they are not fake-quantized inside the recurrent
  forward pass.

## CICIoT CSV Training

The CICIoT path reads `client_0.csv ... client_9.csv` and `global_test.csv`
directly into RAM. The CSV loader builds the global `label_map`, fits a
`StandardScaler` on client CSVs, tracks client sample counts, and returns normal
PyTorch `DataLoader` objects.

Train locally with:

```bash
python train.py --config configs/ciciot_csv_qat.yaml
```

If the CSV dataset is somewhere else, override the config path:

```bash
python train.py --config configs/ciciot_csv_qat.yaml --data-dir /path/to/03_experiments
```

On Kaggle, point `data_dir` at the read-only dataset directory that contains
`client_0.csv ... client_9.csv` and `global_test.csv`, then train directly:

```bash
python train.py --config configs/ciciot_csv_qat_kaggle.yaml
```

or:

```bash
python train.py --config configs/ciciot_csv_qat.yaml \
  --data-dir /kaggle/input/<dataset-dir>/03_experiments
```

## Training Flow

Each round selects clients, trains each local copy for `tau_epochs`, enables
fake-quantized weights for `qat_epochs`, quantizes `w_local - w_global`, and
applies weighted FedAvg on the server.

Important reproduction rule: QAT-FL sends `Q(w_local - w_global)`, not the full
local model.
