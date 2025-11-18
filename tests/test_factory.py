import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.factory import get_model, get_optimizer
from src.models.unet import UNet
from src.models.architectures import ConvexNet, MambaIRv2

def test_get_model_unet():
    config = {
        'architecture': 'unet',
        'in_channels': 3,
        'out_channels': 3,
        'base_channels': 16,
        'activation': 'relu'
    }
    model = get_model(config)
    assert isinstance(model, UNet)
    # Check if activation is correctly set (indirectly via forward pass or inspecting modules)
    # For now just checking instance type is good enough for factory logic

def test_get_model_convexnet():
    config = {
        'architecture': 'convexnet',
        'in_channels': 3,
        'out_channels': 3,
        'base_channels': 16
    }
    model = get_model(config)
    assert isinstance(model, ConvexNet)

def test_get_optimizer_adam():
    model = nn.Linear(10, 10)
    config = {
        'learning_rate': 0.01,
        'optimizer': {
            'type': 'adam',
            'beta1': 0.9,
            'beta2': 0.999
        }
    }
    optimizer = get_optimizer(model, config)
    assert isinstance(optimizer, optim.Adam)
    assert optimizer.defaults['lr'] == 0.01

def test_get_optimizer_adamw():
    model = nn.Linear(10, 10)
    config = {
        'learning_rate': 0.001,
        'optimizer': {
            'type': 'adamw'
        }
    }
    optimizer = get_optimizer(model, config)
    assert isinstance(optimizer, optim.AdamW)

def test_get_optimizer_legacy_string():
    # Test backward compatibility where optimizer might be just a string in some configs
    model = nn.Linear(10, 10)
    config = {
        'learning_rate': 0.001,
        'optimizer': 'sgd'
    }
    optimizer = get_optimizer(model, config)
    assert isinstance(optimizer, optim.SGD)
