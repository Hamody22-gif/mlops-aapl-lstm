from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
import sys
import os
import joblib # NEW: For loading the scaler
import numpy as np # NEW: For array manipulation

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import EnhancedLSTM
from src.config import ModelConfig

app = FastAPI(title="Stock Prediction API", description="API for predicting stock prices using LSTM", version="1.0")

# --- GLOBAL VARIABLES ---
model = None
scaler = None # NEW: The translator

# --- LOAD RESOURCES ---
try:
    print("Loading resources...")
    
    # 1. Load Model
    config = ModelConfig()
    model = EnhancedLSTM(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout
    )
    model.load_state_dict(torch.load("best_lstm_model.pth", map_location=torch.device('cpu')))
    model.eval()
    print("Model loaded successfully!")
    
    # 2. Load Scaler (NEW)
    scaler = joblib.load("scaler.pkl")
    print("Scaler loaded successfully!")
    
except Exception as e:
    print(f"ERROR: Failed to load resources. {e}")
    model = None
    scaler = None

# --- INPUT SCHEMA ---
class StockRequest(BaseModel):
    features: List[float] # User sends RAW prices (e.g. 150.5, 151.2...)

@app.get("/")
def read_root():
    return {
        "status": "ok", 
        "message": "The API is running!",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.post("/predict")
def predict_stock(request: StockRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or Scaler is not loaded.")
    
    # 0. Get Raw Data
    raw_prices = np.array(request.features).reshape(-1, 1)
    
    # 1. Scale Data (Translator: Dollar -> 0..1)
    # We use the same scaler we saved.
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
    # The scaler expects a 2D array [[val]] to transform back
    predicted_price = scaler.inverse_transform([[predicted_scaled]])[0][0]
    
    return {
        "input_days": len(request.features),
        "prediction_price_usd": float(predicted_price), # The real dollar amount!
        "prediction_raw_scaled": float(predicted_scaled) # For debugging
    }
