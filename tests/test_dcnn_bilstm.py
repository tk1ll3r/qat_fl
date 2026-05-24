import torch

from qat_fl.models.registry import create_model
from qat_fl.quantization.fake_quant import FakeQuantConv1d, prepare_weight_qat


def test_dcnn_bilstm_forward_matches_ids_classes():
    model = create_model("dcnn_bilstm", num_classes=34, input_features=76)
    x = torch.randn(4, 76, 1)

    out = model(x)

    assert out.shape == (4, 34)


def test_dcnn_bilstm_conv1d_is_prepared_for_fake_qat():
    model = create_model("dcnn_bilstm", num_classes=34, input_features=76)

    replaced = prepare_weight_qat(model, num_bits=4)

    assert replaced >= 1
    assert isinstance(model.conv, FakeQuantConv1d)
