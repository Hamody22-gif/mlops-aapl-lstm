# MLflow Integration Guide for Your Notebook

## What We'll Add to Your Notebook

This document shows you EXACTLY what code to add to your `applestock.ipynb` notebook.

---

## Step 1: Add MLflow Import (At the top with other imports)

Find your imports cell and add this line:

```python
import mlflow
import mlflow.pytorch
```

---

## Step 2: Start MLflow Tracking (Before training your model)

Add this code BEFORE you start training:

```python
# Start MLflow experiment
mlflow.set_experiment("Apple_Stock_LSTM")

# Start a new run
mlflow.start_run()

# Log hyperparameters
mlflow.log_param("epochs", 50)  # Change this to your actual epochs
mlflow.log_param("learning_rate", 0.001)  # Change to your actual learning rate
mlflow.log_param("hidden_size", 128)  # Change to your actual hidden size
mlflow.log_param("num_layers", 2)  # Change to your actual number of layers
mlflow.log_param("dropout", 0.2)  # Change to your actual dropout
mlflow.log_param("sequence_length", 60)  # Change to your actual sequence length
```

---

## Step 3: Log Metrics During Training (Inside your training loop)

If you have a training loop, add this inside it:

```python
# After each epoch, log the loss
mlflow.log_metric("train_loss", train_loss, step=epoch)
mlflow.log_metric("val_loss", val_loss, step=epoch)
```

---

## Step 4: Log Final Metrics (After training completes)

After training is done, log your final accuracy:

```python
# Log final metrics
mlflow.log_metric("final_accuracy", 96.41)  # Your actual accuracy
mlflow.log_metric("final_rmse", your_rmse_value)  # If you have RMSE
mlflow.log_metric("final_mae", your_mae_value)  # If you have MAE
```

---

## Step 5: Save the Model (After training)

```python
# Log the model
mlflow.pytorch.log_model(model, "lstm_model")

# End the run
mlflow.end_run()
```

---

## Step 6: View MLflow UI

After running your notebook, open a terminal and run:

```bash
cd H:\Stocks_ML
mlflow ui
```

Then open your browser to: http://localhost:5000

---

## Important Notes:

- ✅ Your existing code stays the same
- ✅ We're just adding tracking on top
- ✅ Replace the values (epochs, learning_rate, etc.) with your actual values
- ✅ The model will still train normally
- ✅ MLflow just records everything

---

## Next Steps:

1. I'll help you identify where to add each piece of code
2. We'll add them one by one
3. You'll run the notebook
4. We'll view the results in MLflow UI

**Ready to continue?**
