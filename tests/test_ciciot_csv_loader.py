import pandas as pd

from qat_fl.data.ciciot_csv_dataloader import CICIoTCSVDataModule


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_ciciot_csv_data_module_builds_metadata_and_batches(tmp_path):
    _write_csv(
        tmp_path / "client_0.csv",
        [
            {"feature_a": 1.0, "feature_b": 2.0, "label": "benign"},
            {"feature_a": 2.0, "feature_b": 3.0, "label": "attack"},
        ],
    )
    _write_csv(
        tmp_path / "client_1.csv",
        [
            {"feature_a": 3.0, "feature_b": 4.0, "label": "benign"},
            {"feature_a": 4.0, "feature_b": 5.0, "label": "attack"},
        ],
    )
    _write_csv(
        tmp_path / "global_test.csv",
        [{"feature_a": 5.0, "feature_b": 6.0, "label": "attack"}],
    )

    data_module = CICIoTCSVDataModule(
        data_dir=tmp_path,
        num_clients=2,
        batch_size=2,
        seed=123,
    )

    assert data_module.num_features == 2
    assert data_module.num_classes == 2
    assert data_module.client_num_samples == {0: 2, 1: 2}

    x, y = next(iter(data_module.get_client_loader(0)))
    assert x.shape == (2, 1, 2)
    assert y.shape == (2,)


def test_ciciot_csv_data_module_supports_label_not_last(tmp_path):
    pd.DataFrame(
        [
            {"label": "benign", "feature_a": 1.0, "feature_b": 2.0},
            {"label": "attack", "feature_a": 2.0, "feature_b": 3.0},
        ],
        columns=["label", "feature_a", "feature_b"],
    ).to_csv(tmp_path / "client_0.csv", index=False)
    pd.DataFrame(
        [{"label": "attack", "feature_a": 3.0, "feature_b": 4.0}],
        columns=["label", "feature_a", "feature_b"],
    ).to_csv(tmp_path / "global_test.csv", index=False)

    data_module = CICIoTCSVDataModule(
        data_dir=tmp_path,
        num_clients=1,
        batch_size=2,
    )

    x, y = next(iter(data_module.get_client_loader(0)))
    assert x.shape == (2, 1, 2)
    assert set(y.tolist()) == {0, 1}
