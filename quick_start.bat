@echo off
REM Quick Start Script for Pothole Detection Training

echo ============================================================
echo POTHOLE DETECTION - COMPLETE TRAINING PIPELINE
echo ============================================================
echo.

REM Check if venv exists
if not exist "venv\" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv venv
    echo Then: venv\Scripts\activate
    echo Then: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

echo [1/4] Organizing Dataset...
echo ============================================================
python scripts\organize_dataset.py
if errorlevel 1 (
    echo ERROR: Dataset organization failed!
    pause
    exit /b 1
)

echo.
echo [2/4] Training Model (this will take 1-3 hours)...
echo ============================================================
echo Training with: YOLO11-Small, Baseline hyperparameters
echo.
python scripts\train_model.py --model small --hyperparams baseline
if errorlevel 1 (
    echo ERROR: Training failed!
    pause
    exit /b 1
)

echo.
echo [3/4] Evaluating Model...
echo ============================================================
python scripts\evaluate_model.py
if errorlevel 1 (
    echo ERROR: Evaluation failed!
    pause
    exit /b 1
)

echo.
echo [4/4] Launching Application...
echo ============================================================
python app\main_enhanced.py

echo.
echo ============================================================
echo COMPLETE!
echo ============================================================
pause
