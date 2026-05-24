# QAT-FL TMLCN 2025 Rebuild

This repository is a clean rebuild scaffold for:

**Communication Efficient Federated Learning With Quantization-Aware Training Design**  
DOI: `10.1109/TMLCN.2025.3635050`

The implementation is intentionally modular. The core FL loop depends on small
interfaces for models, quantizers, client update protocols, and aggregation, so
new techniques such as Lloyd-Max non-uniform quantization, adaptive bits,
pruning, distillation, or client-selection policies can be added without
rewriting the trainer.

## Implemented Baseline Surface

- FedAvg full precision via unquantized model deltas.
- PTQ-FL style communication quantization over **model delta**.
- QAT-FL local flow: `tau` normal local epochs, then `M` fake-quantized QAT epochs.
- Uniform symmetric per-tensor quantization with communication-bit accounting.
- Adaptive bit scheduler from the paper note:
  `num_bits_k = ceil(log2(sqrt(initial_loss / current_loss) * 2^initial_bits))`.
- Lloyd-Max/K-means non-uniform quantizer module as an extension point.
- Synthetic smoke dataset so the repo can run without downloads.
- DCNN-BiLSTM IDS model option ported from the public Keras notebook for
  "DCNNBiLSTM: An Efficient Hybrid Deep Learning-Based Intrusion Detection System".

## Layout

```text
qat_fl/
  data/           dataset partition helpers
  fl/             client, server, FedAvg, trainer
  models/         model registry and CNNs
  quantization/   uniform, fake quant, Lloyd-Max, adaptive bits
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
- The repo keeps federated QAT/PTQ training instead of the notebook's centralized
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

## CICIoT CSV Conversion

For large CICIoT2023 client CSVs, convert once to scaled `.npy` arrays:

```bash
python scripts/prepare_ciciot_npy_dataset.py \
  --input-dir /path/to/03_experiments \
  --output-dir data/03_experiments_npy \
  --num-clients 10
```

The converter follows `/mnt/c/Thuong/Research/feder_learn/scripts/data_loader.py`:
it builds `label_map`, `StandardScaler`, sample counts, and class weights from
`client_*.csv`, then transforms both clients and `global_test.csv`. Train with:

```bash
python train.py --config configs/ciciot_npy_qat.yaml
```

## Paper-Aligned Experiment Order

1. `fedavg_fp32`: train local model and send full-precision delta.
2. `ptq_fl`: train full precision for `tau + M` epochs, then quantize delta.
3. `qat_fl`: train `tau` normal epochs, then `M` fake-quantized epochs, then quantize delta.
4. `qat_fl_nonuniform`: swap quantizer to Lloyd-Max.
5. `qat_fl_adaptive`: enable adaptive bits and plot metrics by cumulative communicated bits.

Important reproduction rule: QAT-FL sends `Q(w_local - w_global)`, not the full
local model.
