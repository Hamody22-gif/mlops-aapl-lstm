
import torch
import numpy as np
import pandas as pd
import yfinance as yf
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

class StockDataset(Dataset):
    """Custom Dataset for stock price sequences"""
    def __init__(self, data, seq_len=60):
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        return (self.data[idx:idx+self.seq_len],
                self.data[idx+self.seq_len])

def get_data(ticker='AAPL', start='2019-01-01', end=None):
    """Download stock data using yfinance"""
    if end is None:
        end = datetime.now()
    
    print(f'Downloading {ticker} data...')
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

def prepare_dataloaders(df, seq_len=60, batch_size=32):
    """Prepare DataLoaders for training, validation, and testing"""
    # Prepare data
    data = pd.DataFrame(df['Close'].squeeze())
    dataset = data.values
    
    # Split: 70% train, 15% val, 15% test
    train_size = int(len(dataset) * 0.70)
    val_size = int(len(dataset) * 0.85)
    
    # Scale data - FIT ONLY on training data to prevent data leakage
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(dataset[:train_size])  # Fit only on training data
    scaled_data = scaler.transform(dataset)  # Transform the entire dataset
    
    # Create datasets with correct offsets to prevent data leakage/overlap issues
    train_dataset = StockDataset(scaled_data[:train_size], seq_len=seq_len)
    val_dataset = StockDataset(scaled_data[train_size-seq_len:val_size], seq_len=seq_len)
    test_dataset = StockDataset(scaled_data[val_size-seq_len:], seq_len=seq_len)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Data prepared: Train {len(train_loader)} batches, Val {len(val_loader)} batches, Test {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader, scaler
