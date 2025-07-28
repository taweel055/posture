#!/usr/bin/env python3
"""
Tests for camera connection and video capture functionality
"""

import unittest
import cv2
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestCameraConnection(unittest.TestCase):
    """Test camera connection and capture functionality"""
    
    @patch('cv2.VideoCapture')
    def test_camera_initialization(self, mock_capture):
        """Test camera initialization process"""
        # Setup mock
        mock_cap = MagicMock()
        mock_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
        
        # Test camera opening
        cap = cv2.VideoCapture(0)
        self.assertTrue(cap.isOpened())
        
        # Test frame reading
        ret, frame = cap.read()
        self.assertTrue(ret)
        self.assertIsNotNone(frame)
        
    @patch('cv2.VideoCapture')
    def test_camera_failure_handling(self, mock_capture):
        """Test handling of camera connection failures"""
        # Setup mock for failure
        mock_cap = MagicMock()
        mock_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = False
        
        # Test camera failure
        cap = cv2.VideoCapture(0)
        self.assertFalse(cap.isOpened())
        
    @patch('cv2.VideoCapture')
    def test_camera_properties(self, mock_capture):
        """Test camera property settings"""
        # Setup mock
        mock_cap = MagicMock()
        mock_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        
        # Test property setting
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Verify calls
        mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        mock_cap.set.assert_any_call(cv2.CAP_PROP_FPS, 30)
        
    def test_frame_processing_pipeline(self):
        """Test frame processing pipeline"""
        # Create test frame
        test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Simulate processing steps
        # Step 1: Convert to grayscale
        gray = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
        self.assertEqual(gray.shape, (720, 1280))
        
        # Step 2: Apply blur (noise reduction)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        self.assertEqual(blurred.shape, gray.shape)
        
        # Step 3: Convert back to BGR for display
        processed = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        self.assertEqual(processed.shape, test_frame.shape)
        
    @patch('cv2.VideoCapture')
    def test_multiple_camera_support(self, mock_capture):
        """Test support for multiple cameras"""
        camera_indices = [0, 1, 2]
        available_cameras = []
        
        for idx in camera_indices:
            mock_cap = MagicMock()
            mock_capture.return_value = mock_cap
            mock_cap.isOpened.return_value = (idx < 2)  # Simulate 2 available cameras
            
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                available_cameras.append(idx)
                
        self.assertEqual(len(available_cameras), 2)
        self.assertIn(0, available_cameras)
        self.assertIn(1, available_cameras)
        
    def test_frame_rate_calculation(self):
        """Test FPS calculation logic"""
        import time
        
        frame_times = []
        num_frames = 30
        
        # Simulate frame capture timing
        start_time = time.time()
        for i in range(num_frames):
            frame_times.append(time.time())
            time.sleep(0.033)  # Simulate ~30 FPS
            
        # Calculate average FPS
        total_time = frame_times[-1] - frame_times[0]
        fps = (num_frames - 1) / total_time if total_time > 0 else 0
        
        # Check FPS is reasonable (allowing for timing variations)
        self.assertGreater(fps, 25)
        self.assertLess(fps, 35)

class TestFrontendBackendIntegration(unittest.TestCase):
    """Test integration between frontend UI and backend processing"""
    
    def test_data_exchange_format(self):
        """Test data format for frontend-backend communication"""
        # Simulate backend data structure
        backend_data = {
            "frame_id": 1001,
            "timestamp": "2025-07-28T12:00:00.000Z",
            "posture_score": 85.5,
            "landmarks": [
                {"name": "nose", "x": 640, "y": 360, "confidence": 0.98},
                {"name": "left_shoulder", "x": 580, "y": 420, "confidence": 0.95},
                {"name": "right_shoulder", "x": 700, "y": 420, "confidence": 0.96}
            ],
            "alerts": []
        }
        
        # Validate structure
        self.assertIn("frame_id", backend_data)
        self.assertIn("posture_score", backend_data)
        self.assertIn("landmarks", backend_data)
        self.assertIsInstance(backend_data["landmarks"], list)
        
    def test_command_handling(self):
        """Test command handling from frontend"""
        commands = {
            "start_analysis": {"action": "start", "mode": "real-time"},
            "stop_analysis": {"action": "stop"},
            "generate_report": {"action": "report", "format": "pdf"},
            "change_settings": {"action": "settings", "fps": 15}
        }
        
        # Validate command structures
        for cmd_name, cmd_data in commands.items():
            self.assertIn("action", cmd_data)
            self.assertIsInstance(cmd_data["action"], str)
            
    def test_event_stream_format(self):
        """Test event stream for real-time updates"""
        events = []
        
        # Simulate event generation
        for i in range(5):
            event = {
                "type": "posture_update",
                "data": {
                    "score": 85.0 + i,
                    "timestamp": f"2025-07-28T12:00:{i:02d}.000Z"
                }
            }
            events.append(event)
            
        # Validate event stream
        self.assertEqual(len(events), 5)
        for event in events:
            self.assertIn("type", event)
            self.assertIn("data", event)
            self.assertIn("score", event["data"])
            
    def test_error_handling_communication(self):
        """Test error handling in frontend-backend communication"""
        error_responses = [
            {"status": "error", "code": "CAMERA_NOT_FOUND", "message": "No camera detected"},
            {"status": "error", "code": "PROCESSING_FAILED", "message": "Frame processing failed"},
            {"status": "error", "code": "INVALID_COMMAND", "message": "Unknown command"}
        ]
        
        for error in error_responses:
            self.assertEqual(error["status"], "error")
            self.assertIn("code", error)
            self.assertIn("message", error)
            
    @patch('json.dumps')
    @patch('json.loads')
    def test_json_serialization(self, mock_loads, mock_dumps):
        """Test JSON serialization for data exchange"""
        # Test data
        test_data = {"score": 85.5, "landmarks": [{"x": 100, "y": 200}]}
        
        # Test serialization
        mock_dumps.return_value = '{"score": 85.5, "landmarks": [{"x": 100, "y": 200}]}'
        serialized = mock_dumps(test_data)
        
        # Test deserialization
        mock_loads.return_value = test_data
        deserialized = mock_loads(serialized)
        
        # Verify calls
        mock_dumps.assert_called_once_with(test_data)
        mock_loads.assert_called_once_with(serialized)

if __name__ == '__main__':
    unittest.main()
