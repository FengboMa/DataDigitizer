#!/bin/bash

# Get the directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

VENV_PATH="./PlotDigitizor"

# Check if venv exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment 'PlotDigitizor' not found."
    echo "Please run: python3 -m venv PlotDigitizor && ./PlotDigitizor/bin/pip install -r requirements.txt"
    exit 1
fi

echo "------------------------------------------"
echo "📊 Raman Data Digitizer Launcher"
echo "------------------------------------------"
echo "1) Run Desktop App (Tkinter)"
echo "2) Run Web App (Streamlit)"
echo "q) Quit"
echo "------------------------------------------"
read -p "Select an option [1-2]: " choice

case $choice in
    1)
        echo "🚀 Starting Tkinter App..."
        $VENV_PATH/bin/python3 main.py
        ;;
    2)
        if [ ! -f "streamlit_app.py" ]; then
            echo "❌ streamlit_app.py not found in this branch."
            echo "Make sure you are on the 'feature/streamlit-port' branch."
            exit 1
        fi
        echo "🚀 Starting Streamlit Server..."
        $VENV_PATH/bin/streamlit run streamlit_app.py
        ;;
    q|Q)
        exit 0
        ;;
    *)
        echo "Invalid option."
        ;;
esac
