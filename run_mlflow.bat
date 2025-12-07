@echo off
setlocal
echo Starting MLflow UI...
echo.
echo Checking for Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not found in your PATH. 
    echo Please make sure Python is installed and added to your PATH.
    pause
    exit /b
)

echo Found Python. Launching MLflow UI...
echo Once started, open your web browser to http://localhost:5000
echo Press Ctrl+C to stop the server when you are done.
echo.
cd /d "%~dp0"
:: Point MLflow to the notebooks directory where the data is stored
python -m mlflow ui --backend-store-uri notebooks/mlruns
if %errorlevel% neq 0 (
    echo.
    echo Failed to start MLflow UI. 
    echo Make sure mlflow is installed: pip install mlflow
)
pause
