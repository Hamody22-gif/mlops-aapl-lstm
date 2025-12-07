
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow

def evaluate_model(model, loader, scaler, device, dataset_name="Test"):
    """Evaluate model and log results to MLflow (no plots)"""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            predictions.extend(out.cpu().numpy())
            actuals.extend(y.cpu().numpy())
            
    predictions = np.array(predictions).reshape(-1, 1)
    actuals = np.array(actuals).reshape(-1, 1)
    
    # Inverse transform
    predictions_inv = scaler.inverse_transform(predictions)
    actuals_inv = scaler.inverse_transform(actuals)
    
    # Metrics
    mse = mean_squared_error(actuals_inv, predictions_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals_inv, predictions_inv)
    r2 = r2_score(actuals_inv, predictions_inv)
    
    print(f"\n{dataset_name} Metrics:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE:  {mae:.4f}")
    print(f"   R2:   {r2:.4f}")
    
    # Log metrics to MLflow
    mlflow.log_metrics({
        f"{dataset_name.lower()}_rmse": rmse,
        f"{dataset_name.lower()}_mae": mae,
        f"{dataset_name.lower()}_r2": r2
    })
    
    return rmse, mae, r2
