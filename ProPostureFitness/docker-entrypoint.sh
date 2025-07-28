#!/bin/bash
# Docker entry point script for FitLifePosture

if [ "$INTERFACE_MODE" = "web" ]; then
    echo "🌐 Starting FitLifePosture Web Interface..."
    echo "📡 Access at: http://localhost:8080"
    python web_interface.py
else
    echo "🖥️ Starting FitLifePosture GUI..."
    python final_working_app.py
fi
