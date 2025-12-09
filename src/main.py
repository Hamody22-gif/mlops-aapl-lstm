
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
from logger_config import setup_logger

from config import AppConfig

def main():
    # Load Configuration
    config = AppConfig()
    
    # Setup logger first
    logger = setup_logger()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 1. Load Data
    logger.info(f"Configuration: {config.model_dump_json(indent=2)}")
    df = get_data(ticker=config.data.ticker)
    
    # 2. Prepare Loaders
    train_loader, val_loader, test_loader, scaler = prepare_dataloaders(
        df, 
        seq_len=config.train.sequence_length, 
        batch_size=config.train.batch_size
    )
    
    # 3. Init Model
    model = EnhancedLSTM(
        input_size=config.model.input_size, 
        hidden_size=config.model.hidden_size, 
        num_layers=config.model.num_layers, 
        dropout=config.model.dropout
    ).to(device)
    
    # 4. Train
    logger.info("Starting Training Pipeline...")
    model, _, _ = train_model(
        model, 
        train_loader, 
        val_loader, 
        config=config.train, # Pass config object
        device=device
    )
    
    # 5. Evaluate
    logger.info("Evaluating on Test Set...")
    evaluate_model(model, test_loader, scaler, device, dataset_name="Test")

if __name__ == "__main__":
    main()
