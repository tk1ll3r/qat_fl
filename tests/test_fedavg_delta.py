import torch

from qat_fl.fl.fedavg import aggregate_deltas


def test_aggregate_deltas_weights_by_client_samples():
    ref = {"w": torch.tensor([1.0, 2.0])}
    deltas = [{"w": torch.tensor([1.0, 0.0])}, {"w": torch.tensor([3.0, 2.0])}]

    out = aggregate_deltas(ref, deltas, [1, 3])

    assert torch.allclose(out["w"], torch.tensor([3.5, 3.5]))
