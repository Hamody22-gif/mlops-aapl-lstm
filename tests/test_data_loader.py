
import pytest
import torch
import pandas as pd
import numpy as np
from src.data_loader import StockDataset, prepare_dataloaders
from src.config import TrainConfig

def test_stock_dataset_creation():
    """Test if dataset handles shapes and sequencing correctly"""
    # Create fake data (100 days of prices)
    data = np.random.rand(100, 1)
    
    # Create dataset with sequence length 10
    dataset = StockDataset(data, seq_len=10)
    
    # Expected length: total (100) - seq_len (10) = 90
    assert len(dataset) == 90
    
    # Check item shape
    x, y = dataset[0]
    # Input should be (seq_len, features) -> (10, 1)
    assert x.shape == (10, 1)
    # Target should be (features,) -> (1,)
    assert y.shape == (1,)

def test_stock_dataset_types():
    """Test if dataset returns PyTorch tensors"""
    data = np.random.rand(50, 1)
    dataset = StockDataset(data, seq_len=5)
    
    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)

def test_prepare_dataloaders():
    """Test full data loading pipeline"""
    # Create fake DataFrame simulating 1000 days of stock prices
    dates = pd.date_range(start='2020-01-01', periods=1000)
    df = pd.DataFrame({
        'Close': np.random.rand(1000)
    }, index=dates)
    
    # Config
    seq_len = 10
    batch_size = 32
    
    train_loader, val_loader, test_loader, scaler = prepare_dataloaders(
        df, seq_len=seq_len, batch_size=batch_size
    )
    
    # Check if we got all objects
    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None
    assert scaler is not None
    
    # Check data split logic (approximate sizes check)
    # Total samples available = 1000 - 10 = 990
    # Train is 70% of 1000 = 700. Dataset size ~ 700-seq_len
    # Just checking if they are not empty is good enough for basic smoke test
    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0
    
    # Check batch shape from loader
    x_batch, y_batch = next(iter(train_loader))
    assert x_batch.shape == (batch_size, seq_len, 1)
    assert y_batch.shape == (batch_size, 1)
