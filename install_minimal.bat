@echo off
echo ===================================================
echo DocuMind Minimal Installation Script
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

echo Installing minimal required packages...
pip install --upgrade pip setuptools wheel
pip install fastapi uvicorn python-docx pypdf2 numpy faiss-cpu requests streamlit

echo ===================================================
echo Minimal installation completed!
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
echo This is a minimal installation with basic functionality.
echo For full functionality, run install.bat followed by install_advanced.bat
echo ===================================================
pause