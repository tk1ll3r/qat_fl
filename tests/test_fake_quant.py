import torch

from qat_fl.quantization.fake_quant import fake_quantize_ste


def test_fake_quant_ste_keeps_gradient_path():
    x = torch.tensor([0.1, 0.2, 0.9], requires_grad=True)
    y = fake_quantize_ste(x, num_bits=2).sum()
    y.backward()

    assert torch.allclose(x.grad, torch.ones_like(x))

