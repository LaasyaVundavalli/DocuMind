#!/bin/bash

echo "==================================================="
echo "DocuMind Installation Script"
echo "==================================================="

echo "Creating Python virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "Failed to create virtual environment."
    echo "Please make sure Python 3.10+ is installed."
    exit 1
fi

echo "Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment."
    exit 1
fi

echo "Installing backend dependencies..."
pip install -r backend/requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install backend dependencies."
    exit 1
fi

echo "Installing frontend dependencies..."
pip install -r frontend/requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install frontend dependencies."
    exit 1
fi

echo "==================================================="
echo "Installation completed successfully!"
echo "==================================================="
echo
echo "To start the application:"
echo
echo "1. Open two separate terminal windows"
echo "2. In the first window, run:"
echo "   source venv/bin/activate"
echo "   uvicorn backend.app:app --reload"
echo
echo "3. In the second window, run:"
echo "   source venv/bin/activate"
echo "   streamlit run frontend/app.py"
echo
echo "==================================================="