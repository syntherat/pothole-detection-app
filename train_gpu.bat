@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
python scripts/train_model.py --model medium --hyperparams baseline
pause
