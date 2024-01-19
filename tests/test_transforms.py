import pytest
import torch
from libface.transforms import SquarePad


@pytest.mark.unit
@pytest.mark.transforms
def test_square():
    transform = SquarePad()
    tensor = torch.zeros(1, 3, 200, 100)
    output = transform(tensor)
    assert output.shape[2] == output.shape[3]
