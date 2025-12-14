@echo off
echo Starting Stock Prediction API...
echo ---------------------------------------------------
echo API will be available at: http://127.0.0.1:8000
echo Documentation available at: http://127.0.0.1:8000/docs
echo ---------------------------------------------------
echo Press Ctrl+C to stop the server
echo.
python -m uvicorn api.main:app --reload
pause
