@echo off
echo Starting Zorbite ML Prediction Server...

REM Create logs directory if it doesn't exist
if not exist logs mkdir logs

REM Create data directory if it doesn't exist
if not exist data mkdir data

REM Create models directory if it doesn't exist
if not exist models mkdir models

REM Activate Python environment if needed (uncomment and modify if using virtual environment)
REM call venv\Scripts\activate.bat

REM Run the Zorbite ML server
python python_scripts\zorbite_ml.py

pause 