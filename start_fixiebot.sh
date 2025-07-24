#!/bin/bash

echo ""
echo "========================================"
echo "   FixieBot - AI Ticket Fix Predictor"
echo "========================================"
echo ""
echo "Starting FixieBot..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "ERROR: Python 3.8 or higher is required"
    echo "Current version: $python_version"
    exit 1
fi

# Install dependencies if needed
echo "Checking dependencies..."
pip3 install -r requirements.txt > /dev/null 2>&1

# Start the application
echo "Starting FixieBot server..."
echo ""
echo "The application will be available at: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""

python3 run.py 