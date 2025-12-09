
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
import os
from loguru import logger
from config import TrainConfig

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        predictions = model(x)
        loss = criterion(predictions, y)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            predictions = model(x)
            loss = criterion(predictions, y)
            total_loss += loss.item()
    
    return total_loss / len(loader)

def train_model(model, train_loader, val_loader, config: TrainConfig, device='cpu'):
    """Full training routine with MLflow tracking"""
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    logger.info(f"Starting training on {device}...")
    
    # MLflow Tracking
    mlflow.set_experiment(config.experiment_name)
    
    # End any active run
    if mlflow.active_run():
        mlflow.end_run()
        
    with mlflow.start_run():
        # Log Parameters
        mlflow.log_params({
            'epochs': config.epochs,
            'learning_rate': config.learning_rate,
            'hidden_size': model.hidden_size,
            'num_layers': model.num_layers,
            'optimizer': 'Adam',
            'scheduler': 'ReduceLROnPlateau'
        })
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(config.epochs):
            # Train
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            train_losses.append(train_loss)
            
            # Validate
            val_loss = validate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            
            # Step scheduler
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_lstm_model.pth')
            
            # Log metrics
            if (epoch + 1) % 10 == 0:
                mlflow.log_metrics({'train_loss': train_loss, 'val_loss': val_loss}, step=epoch)
                logger.info(f"Epoch [{epoch+1}/{config.epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        logger.success(f"Training complete! Best validation loss: {best_val_loss:.6f}")
        
        # Load best model for logging
        model.load_state_dict(torch.load('best_lstm_model.pth'))
        
        # Infer signature and log model
        # Create a dummy input for signature inference
        # Assuming input shape [1, seq_len, 1] - we need to know seq_len, 
        # let's try to get it from the loader or pass it in. 
        # For now, we'll skip signature or try to derive it if possible, 
        # but safely logging without signature is also fine to avoid shape errors if seq_len varies.
        
        try:
            # We'll rely on the user to test, logging without signature for now to be safe
            mlflow.pytorch.log_model(
                model, 
                "lstm_model",
                registered_model_name=config.registered_model_name
            )
            logger.success(f"Model logged to MLflow and registered as '{config.registered_model_name}'")
        except Exception as e:
            logger.error(f"Could not log model to MLflow: {e}")
            
    return model, train_losses, val_losses
