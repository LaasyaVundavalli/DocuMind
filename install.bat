@echo off
echo ===================================================
echo DocuMind Installation Script
echo ===================================================

echo Creating Python virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo Failed to create virtual environment.
    echo Please make sure Python 3.10+ is installed.
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)

echo Installing setuptools and wheel...
pip install --upgrade pip setuptools wheel
if %errorlevel% neq 0 (
    echo Failed to install setuptools and wheel.
    pause
    exit /b 1
)

echo Installing backend dependencies...
pip install -r backend\requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install backend dependencies.
    pause
    exit /b 1
)

echo Installing frontend dependencies...
pip install -r frontend\requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install frontend dependencies.
    pause
    exit /b 1
)

echo ===================================================
echo Installation completed successfully!
echo ===================================================
echo.
echo To start the application:
echo.
echo 1. Open two separate terminal windows
echo 2. In the first window, run:
echo    venv\Scripts\activate
echo    uvicorn backend.app:app --reload
echo.
echo 3. In the second window, run:
echo    venv\Scripts\activate
echo    streamlit run frontend/app.py
echo.
echo ===================================================
pause