@echo off
echo ===================================================
echo DocuMind Advanced Dependencies Installation
echo ===================================================

echo Activating virtual environment...
call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)

echo Installing sentence-transformers...
pip install sentence-transformers
if %errorlevel% neq 0 (
    echo Warning: Failed to install sentence-transformers. Some features may not work.
)

echo Installing chromadb...
pip install chromadb
if %errorlevel% neq 0 (
    echo Warning: Failed to install chromadb. Some features may not work.
)

echo Installing streamlit...
pip install streamlit
if %errorlevel% neq 0 (
    echo Warning: Failed to install streamlit. The frontend will not work.
)

echo ===================================================
echo Advanced dependencies installation completed!
echo ===================================================
echo.
echo To start the application:
echo.
echo 1. Open two separate terminal windows
echo 2. In the first window, run:
echo    start_backend.bat
echo.
echo 3. In the second window, run:
echo    start_frontend.bat
echo.
echo ===================================================
pause