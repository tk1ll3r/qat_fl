import numpy as np

from qat_fl.data.ciciot_npy_dataloader import NpyClassificationDataset


def test_npy_classification_dataset_matches_feder_learn_shape(tmp_path):
    x_path = tmp_path / "client_0_X.npy"
    y_path = tmp_path / "client_0_y.npy"
    np.save(x_path, np.zeros((3, 76), dtype=np.float32))
    np.save(y_path, np.asarray([0, 1, 2], dtype=np.int64))

    dataset = NpyClassificationDataset(x_path, y_path)
    x, y = dataset[0]

    assert x.shape == (1, 76)
    assert y.item() == 0
