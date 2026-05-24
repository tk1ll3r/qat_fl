import torch

from qat_fl.fl.fedavg import aggregate_deltas


def test_aggregate_deltas_uniform_adds_average_delta():
    ref = {"w": torch.tensor([1.0, 2.0])}
    deltas = [{"w": torch.tensor([1.0, 0.0])}, {"w": torch.tensor([3.0, 2.0])}]

    out = aggregate_deltas(ref, deltas, [1, 10], mode="uniform")

    assert torch.allclose(out["w"], torch.tensor([3.0, 3.0]))

