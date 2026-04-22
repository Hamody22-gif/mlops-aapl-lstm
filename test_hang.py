print("1")
import os
print("2")
import torch
print("3")
import joblib
print("4")
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.model import EnhancedLSTM
print("5")
from src.config import ModelConfig
print("6")
config = ModelConfig()
for ticker in ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]:
    print(f"Loading {ticker}")
    model_path = f"models/model_{ticker}.pth"
    scaler_path = f"models/scaler_{ticker}.pkl"
    if os.path.exists(model_path):
        print(f"Loading model {model_path}")
        m = EnhancedLSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout
        )
        m.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        m.eval()
        print(f"Loaded model {model_path}")
    if os.path.exists(scaler_path):
         print(f"Loading scaler {scaler_path}")
         s = joblib.load(scaler_path)
         print(f"Loaded scaler {scaler_path}")
print("DONE!")
