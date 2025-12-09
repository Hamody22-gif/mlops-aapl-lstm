
import pytest
import torch
from src.model import EnhancedLSTM
from src.config import ModelConfig

def test_model_initialization():
    """Test if model initializes with correct parameters"""
    config = ModelConfig(
        input_size=1,
        hidden_size=64,
        num_layers=1,
        dropout=0.2
    )
    
    model = EnhancedLSTM(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout
    )
    
    # Check if attributes are set (assuming they are accessible or checking structure)
    assert model.hidden_size == 64
    assert model.num_layers == 1

def test_model_forward_pass():
    """Test forward pass with dummy data"""
    # Setup
    input_size = 1
    hidden_size = 32
    num_layers = 1
    seq_len = 10
    batch_size = 5
    
    model = EnhancedLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.0
    )
    
    # Create dummy input [Batch, Seq, Features]
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Forward pass
    output = model(x)
    
    # Check output shape: Should be [Batch, 1] for stock prediction
    assert output.shape == (batch_size, 1)

def test_model_device_movement():
    """Test if model moves to device correctly (CPU check)"""
    model = EnhancedLSTM(1, 10, 1, 0)
    device = torch.device('cpu')
    model.to(device)
    
    # Basic check - on CPU usually default, but ensures .to() works without error
    param = next(model.parameters())
    assert param.device.type == 'cpu'
