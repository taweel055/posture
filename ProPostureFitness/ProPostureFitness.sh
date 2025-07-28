#!/bin/bash
# ProPostureFitness v5.0 Launcher for macOS

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtual environment if it exists
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
elif [ -d "$SCRIPT_DIR/../venv" ]; then
    source "$SCRIPT_DIR/../venv/bin/activate"
fi

# Set Python path to include current directory
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Change to application directory
cd "$SCRIPT_DIR"

# Check for camera permissions
echo "🎯 Starting ProPostureFitness v5.0..."
echo "📁 Running from: $SCRIPT_DIR"

# Launch the final working application
python3 final_working_app.py "$@"
