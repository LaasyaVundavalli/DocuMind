@echo off
echo ===================================================
echo DocuMind Installation Verification
echo ===================================================

call venv\Scripts\activate

echo Checking Python version:
python --version

echo.
echo Checking pip version:
pip --version

echo.
echo Checking installed packages:
pip list

echo.
echo Checking if required modules can be imported:

echo.
echo 1. Testing FastAPI import:
python -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')" 2>&1

echo.
echo 2. Testing PyPDF2 import:
python -c "import PyPDF2; print(f'PyPDF2 version: {PyPDF2.__version__}')" 2>&1

echo.
echo 3. Testing NumPy import:
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')" 2>&1

echo.
echo 4. Testing sentence-transformers import (optional):
python -c "try: import sentence_transformers; print('sentence-transformers version: ' + sentence_transformers.__version__); except ImportError: print('sentence-transformers not installed - will use fallback method')" 2^>^&1 && echo SUCCESS || echo FAILED

echo.
echo 5. Testing streamlit import (required for frontend):
python -c "try: import streamlit; print('Streamlit version: ' + streamlit.__version__); except ImportError: print('Streamlit not installed - frontend will not work')" 2^>^&1 && echo SUCCESS || echo FAILED

echo.
echo ===================================================
echo.
echo If any of these checks failed, you may need to install missing packages.
echo Try running install_advanced.bat for additional packages.
echo.
echo ===================================================
pause