import joblib
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import get_data, prepare_dataloaders
from src.config import AppConfig

def main():
    print("Initializing configuration...")
    config = AppConfig()
    
    print(f"Downloading data for {config.data.ticker}...")
    df = get_data(ticker=config.data.ticker)
    
    print("Fitting scaler (this matches the training process)...")
    # We only care about the scaler here, so we ignore the loaders
    _, _, _, scaler = prepare_dataloaders(
        df, 
        seq_len=config.train.sequence_length, 
        batch_size=config.train.batch_size
    )
    
    output_path = "scaler.pkl"
    print(f"Saving scaler to {output_path}...")
    joblib.dump(scaler, output_path)
    print("Done! 'scaler.pkl' is ready for the API.")

if __name__ == "__main__":
    main()
