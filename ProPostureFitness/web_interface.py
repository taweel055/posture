#!/usr/bin/env python3
"""
Web interface wrapper for FitLifePosture
Provides browser-based access to the posture analysis system
Useful for Docker deployments where GUI access is limited
"""

import os
import cv2
import base64
import json
import threading
import time
from flask import Flask, render_template_string, Response, jsonify, request
from datetime import datetime
import numpy as np

# Try to import the unified system
try:
    from unified_posture_system import PostureAnalysisSystem, AnalysisMode
    UNIFIED_SYSTEM_AVAILABLE = True
except ImportError:
    print("Warning: Could not import unified posture system")
    UNIFIED_SYSTEM_AVAILABLE = False

app = Flask(__name__)

# Global variables
camera = None
posture_app = None
analysis_active = False
latest_results = {"score": 0, "timestamp": "", "status": "Not started"}

# HTML template for the web interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>FitLifePosture - Web Interface</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .main-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
        }
        .video-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .stats-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        #videoFeed {
            width: 100%;
            border-radius: 8px;
            background: #000;
        }
        .score-display {
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            margin: 20px 0;
        }
        .score-good { color: #10b981; }
        .score-fair { color: #f59e0b; }
        .score-poor { color: #ef4444; }
        .controls {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }
        button {
            flex: 1;
            padding: 12px 24px;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn-primary {
            background: #667eea;
            color: white;
        }
        .btn-primary:hover {
            background: #5a67d8;
        }
        .btn-secondary {
            background: #e5e7eb;
            color: #374151;
        }
        .btn-secondary:hover {
            background: #d1d5db;
        }
        .status {
            padding: 8px 16px;
            border-radius: 20px;
            display: inline-block;
            font-size: 14px;
            font-weight: 500;
        }
        .status-active {
            background: #d1fae5;
            color: #065f46;
        }
        .status-inactive {
            background: #fee2e2;
            color: #991b1b;
        }
        .metric {
            margin: 15px 0;
            padding: 15px;
            background: #f9fafb;
            border-radius: 8px;
        }
        .metric-label {
            font-size: 14px;
            color: #6b7280;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: 600;
            color: #1f2937;
        }
        @media (max-width: 768px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏃‍♂️ FitLifePosture</h1>
            <p>Professional Posture Analysis System - Web Interface</p>
        </div>
        
        <div class="main-grid">
            <div class="video-container">
                <h2>Live Camera Feed</h2>
                <img id="videoFeed" src="/video_feed" alt="Video stream">
                <div class="controls">
                    <button class="btn-primary" onclick="toggleAnalysis()">
                        <span id="toggleText">Start Analysis</span>
                    </button>
                    <button class="btn-secondary" onclick="captureSnapshot()">
                        📸 Capture
                    </button>
                    <button class="btn-secondary" onclick="downloadReport()">
                        📊 Report
                    </button>
                </div>
            </div>
            
            <div class="stats-container">
                <h2>Analysis Results</h2>
                <div class="status status-inactive" id="statusBadge">
                    Inactive
                </div>
                
                <div class="score-display score-fair" id="scoreDisplay">
                    --
                </div>
                
                <div class="metric">
                    <div class="metric-label">Posture Rating</div>
                    <div class="metric-value" id="postureRating">-</div>
                </div>
                
                <div class="metric">
                    <div class="metric-label">Last Updated</div>
                    <div class="metric-value" id="lastUpdated">-</div>
                </div>
                
                <div class="metric">
                    <div class="metric-label">Recommendations</div>
                    <div id="recommendations" style="margin-top: 10px; font-size: 14px; line-height: 1.6;">
                        Start analysis to see recommendations
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let analysisActive = false;
        
        function toggleAnalysis() {
            analysisActive = !analysisActive;
            fetch('/toggle_analysis', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    document.getElementById('toggleText').textContent = 
                        data.active ? 'Stop Analysis' : 'Start Analysis';
                    updateStatus(data.active);
                });
        }
        
        function updateStatus(active) {
            const badge = document.getElementById('statusBadge');
            if (active) {
                badge.textContent = 'Active';
                badge.className = 'status status-active';
            } else {
                badge.textContent = 'Inactive';
                badge.className = 'status status-inactive';
            }
        }
        
        function updateResults() {
            fetch('/get_results')
                .then(response => response.json())
                .then(data => {
                    const score = data.score;
                    const scoreDisplay = document.getElementById('scoreDisplay');
                    
                    // Update score
                    scoreDisplay.textContent = score > 0 ? score.toFixed(0) : '--';
                    
                    // Update score color
                    if (score >= 80) {
                        scoreDisplay.className = 'score-display score-good';
                    } else if (score >= 60) {
                        scoreDisplay.className = 'score-display score-fair';
                    } else {
                        scoreDisplay.className = 'score-display score-poor';
                    }
                    
                    // Update rating
                    let rating = '-';
                    if (score >= 90) rating = 'Excellent';
                    else if (score >= 80) rating = 'Good';
                    else if (score >= 70) rating = 'Fair';
                    else if (score >= 60) rating = 'Needs Improvement';
                    else if (score > 0) rating = 'Poor';
                    document.getElementById('postureRating').textContent = rating;
                    
                    // Update timestamp
                    document.getElementById('lastUpdated').textContent = 
                        data.timestamp || '-';
                    
                    // Update recommendations
                    let recommendations = '';
                    if (score > 0 && score < 90) {
                        if (score < 70) {
                            recommendations = '• Adjust your sitting position<br>' +
                                            '• Keep your back straight<br>' +
                                            '• Align your head with spine';
                        } else {
                            recommendations = '• Minor adjustments needed<br>' +
                                            '• Maintain current posture<br>' +
                                            '• Take regular breaks';
                        }
                    } else if (score >= 90) {
                        recommendations = '• Excellent posture!<br>' +
                                        '• Keep up the good work<br>' +
                                        '• Remember to stretch';
                    }
                    document.getElementById('recommendations').innerHTML = 
                        recommendations || 'Start analysis to see recommendations';
                });
        }
        
        function captureSnapshot() {
            fetch('/capture_snapshot', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    alert('Snapshot captured: ' + data.filename);
                });
        }
        
        function downloadReport() {
            window.open('/download_report', '_blank');
        }
        
        // Update results every 2 seconds
        setInterval(updateResults, 2000);
        
        // Initial update
        updateResults();
    </script>
</body>
</html>
'''

def generate_frames():
    """Generate frames for video streaming"""
    global camera, posture_app
    
    while True:
        if camera is None:
            # Return a placeholder image
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Camera not available", (150, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _, buffer = cv2.imencode('.jpg', placeholder)
            frame = buffer.tobytes()
        else:
            success, frame = camera.read()
            if not success:
                continue
            
            # Analyze posture if active
            if analysis_active and posture_app:
                frame = posture_app.analyze_posture(frame)
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_analysis', methods=['POST'])
def toggle_analysis():
    """Toggle posture analysis"""
    global analysis_active
    analysis_active = not analysis_active
    return jsonify({"active": analysis_active})

@app.route('/get_results')
def get_results():
    """Get latest analysis results"""
    global latest_results
    latest_results["timestamp"] = datetime.now().strftime("%H:%M:%S")
    return jsonify(latest_results)

@app.route('/capture_snapshot', methods=['POST'])
def capture_snapshot():
    """Capture a snapshot"""
    filename = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    # In real implementation, save the current frame
    return jsonify({"filename": filename})

@app.route('/download_report')
def download_report():
    """Generate and download report"""
    # In real implementation, generate PDF report
    return "Report generation not implemented in demo"

def init_camera():
    """Initialize camera"""
    global camera
    try:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("✅ Camera initialized")
    except Exception as e:
        print(f"❌ Camera initialization failed: {e}")
        camera = None

def init_posture_app():
    """Initialize posture analysis app"""
    global posture_app
    try:
        if UNIFIED_SYSTEM_AVAILABLE:
            posture_app = PostureAnalysisSystem(mode=AnalysisMode.ADVANCED)
            print("✅ Posture app initialized")
    except Exception as e:
        print(f"❌ Posture app initialization failed: {e}")
        posture_app = None

if __name__ == '__main__':
    print("🌐 Starting FitLifePosture Web Interface...")
    init_camera()
    init_posture_app()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=8080, debug=False)
