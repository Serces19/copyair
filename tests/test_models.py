"""
Unit tests for all model architectures in CopyAir.
Verifies instantiation, forward pass with arbitrary dimensions, and output shape/value ranges.
"""

import pytest
import torch
from src.models.factory import get_model


@pytest.fixture
def dummy_input():
    return torch.randn(1, 3, 128, 128)


@pytest.mark.parametrize("arch,size", [
    ("nafnet", "small"),
    ("nafnet", "base"),
    ("scope_unet", "small"),
    ("scope_unet", "base"),
    ("restormer", "tiny"),
    ("restormer", "small"),
    ("mambair", "tiny"),
    ("mambair", "base"),
    ("convnext", "nano"),
    ("residual_unet", "base"),
    ("smart_unet", "base"),
    ("basic_unet", "base"),
])
def test_model_forward_pass(arch, size, dummy_input):
    config = {
        "architecture": arch,
        "size": size,
        "in_channels": 3,
        "out_channels": 3,
        "base_channels": 32,
        "dropout_p": 0.0
    }
    model = get_model(config)
    model.eval()

    with torch.no_grad():
        output = model(dummy_input)

    assert isinstance(output, torch.Tensor), f"{arch} did not return a Tensor"
    assert output.shape == dummy_input.shape, f"{arch} output shape {output.shape} != input shape {dummy_input.shape}"
    assert not torch.isnan(output).any(), f"{arch} produced NaNs"
    assert not torch.isinf(output).any(), f"{arch} produced Infs"


def test_restormer_arbitrary_resolution():
    model = get_model({"architecture": "restormer", "size": "tiny"})
    model.eval()
    # Test with odd / non-multiple-of-8 dimensions
    x = torch.randn(1, 3, 137, 219)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 3, 137, 219)


def test_nafnet_arbitrary_resolution():
    model = get_model({"architecture": "nafnet", "size": "small"})
    model.eval()
    x = torch.randn(1, 3, 153, 187)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 3, 153, 187)


def test_mambair_forward():
    model = get_model({"architecture": "mambair", "size": "tiny"})
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 3, 64, 64)
