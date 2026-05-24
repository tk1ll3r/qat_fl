import torch

from qat_fl.quantization.uniform import uniform_dequantize, uniform_quantize


def test_uniform_quantizer_round_trips_shape_and_bounds():
    x = torch.linspace(-1.0, 1.0, 17)
    q = uniform_quantize(x, num_bits=4)
    restored = uniform_dequantize(q)

    assert restored.shape == x.shape
    assert q.values.min() >= -8
    assert q.values.max() <= 7
    assert torch.mean(torch.abs(restored - x)) < 0.08

