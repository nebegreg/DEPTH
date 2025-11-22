@echo off
REM Depth Anything v3 GUI Launcher Script (Windows)
REM ================================================
REM Script de lancement simplifié pour l'application GUI (Windows)

echo =========================================
echo   Depth Anything v3 - GUI Professional
echo =========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python 3 not found
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo [OK] Python found
python --version

REM Check if virtual environment exists
if not exist "venv" (
    echo [WARNING] Virtual environment not found
    echo Creating virtual environment...
    python -m venv venv
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment found
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if dependencies are installed
python -c "import PyQt6" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] Dependencies not installed
    echo Installing dependencies... (this may take a few minutes)

    echo Installing PyTorch...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet

    echo Installing GUI dependencies...
    pip install -r requirements_gui.txt --quiet

    REM Install Depth Anything v3
    if exist "Depth-Anything-3-main" (
        echo Installing Depth Anything v3...
        cd Depth-Anything-3-main
        pip install -e . --quiet
        cd ..
    ) else (
        echo [ERROR] Depth-Anything-3-main directory not found
        echo Please extract Depth-Anything-3-main.zip first
        pause
        exit /b 1
    )

    echo [OK] All dependencies installed
) else (
    echo [OK] Dependencies already installed
)

REM Check CUDA availability
echo.
echo Checking GPU availability...
python -c "import torch; print('[OK] CUDA available: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '[WARNING] CUDA not available - running on CPU (slower)')"

echo.
echo =========================================
echo Launching Depth Anything v3 GUI...
echo =========================================
echo.

REM Launch the application
python depth_anything_gui.py

REM Deactivate virtual environment on exit
deactivate
pause
