import torch
from torch.utils.data import DataLoader, TensorDataset

from qat_fl.fl.client import FLClient
from qat_fl.models.registry import create_model
from qat_fl.quantization.uniform import QuantizedTensor


def test_qat_client_sends_quantized_delta_payload():
    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    model = create_model("tiny_cnn", num_classes=10)
    client = FLClient(0, loader, torch.device("cpu"))

    update = client.train_update(model, lr=0.01, tau_epochs=1, qat_epochs=1, strategy="qat_fl", num_bits=4)

    assert any(isinstance(value, QuantizedTensor) for value in update.payload.values())
    assert update.metrics is not None
    assert update.metrics.communicated_bits > 0

