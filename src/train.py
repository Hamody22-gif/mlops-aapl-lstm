
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
import os

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

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, device='cpu', experiment_name="Stock_Price_Prediction_LSTM"):
    """Full training routine with MLflow tracking"""
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    print(f"\nStarting training on {device}...")
    
    # MLflow Tracking
    mlflow.set_experiment(experiment_name)
    
    # End any active run
    if mlflow.active_run():
        mlflow.end_run()
        
    with mlflow.start_run():
        # Log Parameters
        mlflow.log_params({
            'epochs': epochs,
            'learning_rate': lr,
            'hidden_size': model.hidden_size,
            'num_layers': model.num_layers,
            'optimizer': 'Adam',
            'scheduler': 'ReduceLROnPlateau'
        })
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
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
                print(f'Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')
        
        print(f'\nTraining complete! Best validation loss: {best_val_loss:.6f}')
        
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
            mlflow.pytorch.log_model(model, "lstm_model")
            print("Model logged to MLflow")
        except Exception as e:
            print(f"Could not log model to MLflow: {e}")
            
    return model, train_losses, val_losses
