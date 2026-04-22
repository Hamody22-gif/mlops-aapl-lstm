from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
import sys
import os
import joblib 
import numpy as np

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import EnhancedLSTM
from src.config import ModelConfig

app = FastAPI(title="Stock Prediction API", description="API for predicting stock prices using LSTM", version="1.0")

# --- GLOBAL VARIABLES ---
SUPPORTED_TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
models = {}
scalers = {}

# --- LOAD RESOURCES ---
try:
    print("Loading resources for all stocks...")
    config = ModelConfig()
    
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    for ticker in SUPPORTED_TICKERS:
        model_path = os.path.join(root_dir, "models", f"model_{ticker}.pth")
        scaler_path = os.path.join(root_dir, "models", f"scaler_{ticker}.pkl")
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            m = EnhancedLSTM(
                input_size=config.input_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                dropout=config.dropout
            )
            m.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            m.eval()
            
            s = joblib.load(scaler_path)
            
            models[ticker] = m
            scalers[ticker] = s
            print(f"Loaded resources for {ticker}!")
        else:
            print(f"Skipping {ticker}... resources not found yet.")
            
except Exception as e:
    print(f"ERROR: Failed to load resources. {e}")

# --- INPUT SCHEMA ---
class StockRequest(BaseModel):
    ticker: str
    features: List[float]

@app.get("/")
def read_root():
    return {
        "status": "ok", 
        "message": "The Multi-Stock API is running!",
        "loaded_models": list(models.keys())
    }

@app.post("/predict")
def predict_stock(request: StockRequest):
    ticker = request.ticker.upper()
    
    if ticker not in models or ticker not in scalers:
        raise HTTPException(status_code=400, detail=f"Model or Scaler for {ticker} is not available/loaded.")
        
    model = models[ticker]
    scaler = scalers[ticker]
    
    # 0. Get Raw Data
    raw_prices = np.array(request.features).reshape(-1, 1)
    
    # 1. Scale Data (Translator: Dollar -> 0..1)
    scaled_prices = scaler.transform(raw_prices)
    
    # 2. Convert to Tensor
    data = torch.tensor(scaled_prices, dtype=torch.float32)
    
    # 3. Reshape for LSTM (Batch=1, TimeSteps=N, Features=1)
    input_tensor = data.unsqueeze(0) 
    
    # 4. Predict
    with torch.no_grad():
        prediction_tensor = model(input_tensor)
        
    # 5. Extract Result (This is still in 0..1 format)
    predicted_scaled = prediction_tensor.item()
    
    # 6. Inverse Scale (Translator: 0..1 -> Dollar)
    predicted_price = scaler.inverse_transform([[predicted_scaled]])[0][0]
    
    return {
        "ticker": ticker,
        "input_days": len(request.features),
        "prediction_price_usd": float(predicted_price), 
        "prediction_raw_scaled": float(predicted_scaled)
    }
