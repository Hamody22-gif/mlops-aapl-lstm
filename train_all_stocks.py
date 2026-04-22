import argparse
import torch
import os
import sys
import joblib
from loguru import logger

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from data_loader import get_data, prepare_dataloaders
from model import EnhancedLSTM
from train import train_model
from evaluate import evaluate_model
from logger_config import setup_logger
from config import AppConfig

SUPPORTED_STOCKS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

def main():
    # Setup logger
    _ = setup_logger()
    
    # Load default Configuration
    base_config = AppConfig()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Optional: reduce epochs for testing if desired, wait let's keep it as is
    # base_config.train.epochs = 20
    
    for ticker in SUPPORTED_STOCKS:
        logger.info(f"\n{'='*50}\nStarting pipeline for {ticker}\n{'='*50}")
        
        # 1. Update config for current ticker
        config = base_config.model_copy(deep=True)
        config.data.ticker = ticker
        config.train.experiment_name = f"Stock_Prediction_{ticker}"
        config.train.registered_model_name = f"StockPredictor_{ticker}"
        
        # 2. Get Data
        df = get_data(ticker=config.data.ticker)
        
        # 3. Prepare Loaders & Scaler
        train_loader, val_loader, test_loader, scaler = prepare_dataloaders(
            df, 
            seq_len=config.train.sequence_length, 
            batch_size=config.train.batch_size
        )
        
        # Save Scaler!
        scaler_path = f"models/scaler_{ticker}.pkl"
        joblib.dump(scaler, scaler_path)
        logger.success(f"Saved scaler to {scaler_path}")
        
        # 4. Init Model
        model = EnhancedLSTM(
            input_size=config.model.input_size, 
            hidden_size=config.model.hidden_size, 
            num_layers=config.model.num_layers, 
            dropout=config.model.dropout
        ).to(device)
        
        # 5. Train
        model_save_path = f"models/model_{ticker}.pth"
        model, _, _ = train_model(
            model, 
            train_loader, 
            val_loader, 
            config=config.train, 
            device=device,
            model_save_path=model_save_path
        )
        logger.success(f"Saved best model to {model_save_path}")
        
        # 6. Evaluate
        logger.info(f"Evaluating {ticker} on Test Set...")
        evaluate_model(model, test_loader, scaler, device, dataset_name=f"Test_{ticker}")

    logger.success("\nAll models and scalers have been trained and saved successfully!")

if __name__ == "__main__":
    main()
