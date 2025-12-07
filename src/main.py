
import argparse
import torch
import os
import sys

# Add src to path to ensure imports work when running from root or src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import get_data, prepare_dataloaders
from model import EnhancedLSTM
from train import train_model
from evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description='Stock Price Prediction with LSTM')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=60, help='Sequence length')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 1. Load Data
    df = get_data(ticker=args.ticker)
    
    # 2. Prepare Loaders
    train_loader, val_loader, test_loader, scaler = prepare_dataloaders(df, seq_len=args.seq_len, batch_size=args.batch_size)
    
    # 3. Init Model
    model = EnhancedLSTM(
        input_size=1, 
        hidden_size=128, 
        num_layers=2, 
        dropout=0.2
    ).to(device)
    
    # 4. Train
    print("\nStarting Training Pipeline...")
    model, _, _ = train_model(
        model, 
        train_loader, 
        val_loader, 
        epochs=args.epochs, 
        lr=args.lr, 
        device=device
    )
    
    # 5. Evaluate
    print("\nEvaluating on Test Set...")
    evaluate_model(model, test_loader, scaler, device, dataset_name="Test")

if __name__ == "__main__":
    main()
