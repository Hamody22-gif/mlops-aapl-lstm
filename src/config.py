"""
Central Configuration using Pydantic
This module defines all the settings for the project in one place.
It uses Pydantic to validate data types and provide auto-completion.
"""

from pydantic import BaseModel, Field
from typing import Optional

class ModelConfig(BaseModel):
    """Configuration for the LSTM Model"""
    input_size: int = Field(default=1, description="Number of input features (1 for univariate time series)")
    hidden_size: int = Field(default=128, description="Number of hidden units in LSTM layers")
    num_layers: int = Field(default=2, description="Number of stacked LSTM layers")
    dropout: float = Field(default=0.2, description="Dropout probability")

class TrainConfig(BaseModel):
    """Configuration for the Training Loop"""
    batch_size: int = Field(default=32, description="Number of samples per batch")
    epochs: int = Field(default=50, description="Number of training epochs")
    learning_rate: float = Field(default=0.001, description="Learning rate for Adam optimizer")
    sequence_length: int = Field(default=60, description="Length of the input sequence")
    experiment_name: str = Field(default="Stock_Price_Prediction_LSTM", description="Name of the MLflow experiment")
    registered_model_name: str = Field(default="StockPredictor", description="Name of the model in MLflow Registry")

class DataConfig(BaseModel):
    """Configuration for Data Loading"""
    ticker: str = Field(default="AAPL", description="Stock ticker symbol to download")
    start_date: str = Field(default="2019-01-01", description="Start date for data download")
    val_split: float = Field(default=0.15, description="Fraction of data for validation")
    test_split: float = Field(default=0.15, description="Fraction of data for testing")

class AppConfig(BaseModel):
    """Main Application Configuration"""
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    device: str = Field(default="cuda", description="Device to use for training (cuda or cpu)")
