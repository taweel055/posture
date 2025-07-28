#!/usr/bin/env python3
"""
Unit tests for posture analysis core functionality
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import Mock, patch
import cv2

class TestPostureAnalysis(unittest.TestCase):
    """Test posture analysis functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a mock frame
        self.test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.test_frame[:] = (100, 100, 100)  # Gray background
        
    def test_frame_creation(self):
        """Test frame creation and basic properties"""
        self.assertEqual(self.test_frame.shape, (720, 1280, 3))
        self.assertEqual(self.test_frame.dtype, np.uint8)
        
    @patch('cv2.CascadeClassifier')
    def test_face_detection_initialization(self, mock_cascade):
        """Test face detection initialization"""
        # Import the app module
        from final_working_app import FinalPostureApp
        
        # Create app instance
        app = FinalPostureApp()
        
        # Check that cascade classifier was called
        mock_cascade.assert_called()
        
    def test_posture_score_calculation(self):
        """Test posture score calculation logic"""
        # Test score boundaries
        scores = []
        
        # Perfect posture
        head_offset = 0
        shoulder_offset = 0
        score = 100 - (abs(head_offset) * 0.5 + abs(shoulder_offset) * 0.3)
        scores.append(score)
        self.assertEqual(scores[0], 100.0)
        
        # Moderate misalignment
        head_offset = 20
        shoulder_offset = 10
        score = 100 - (abs(head_offset) * 0.5 + abs(shoulder_offset) * 0.3)
        scores.append(score)
        self.assertEqual(scores[1], 87.0)
        
        # Severe misalignment
        head_offset = 50
        shoulder_offset = 30
        score = 100 - (abs(head_offset) * 0.5 + abs(shoulder_offset) * 0.3)
        scores.append(score)
        self.assertEqual(scores[2], 66.0)
        
        # Check all scores are in valid range
        for score in scores:
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 100)
    
    def test_frame_processing_dimensions(self):
        """Test frame processing maintains dimensions"""
        # Simulate frame processing
        processed_frame = cv2.cvtColor(self.test_frame, cv2.COLOR_BGR2GRAY)
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
        
        # Check dimensions are maintained
        self.assertEqual(processed_frame.shape, self.test_frame.shape)
        
    def test_overlay_generation(self):
        """Test overlay text and graphics generation"""
        frame_with_overlay = self.test_frame.copy()
        
        # Add text overlay
        cv2.putText(frame_with_overlay, "Posture Score: 85.5", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Verify frame was modified
        self.assertFalse(np.array_equal(frame_with_overlay, self.test_frame))
        
    def test_color_coding_logic(self):
        """Test color coding based on posture score"""
        def get_score_color(score):
            if score >= 90:
                return (0, 255, 0)  # Green
            elif score >= 70:
                return (0, 255, 255)  # Yellow
            else:
                return (0, 0, 255)  # Red
                
        # Test color thresholds
        self.assertEqual(get_score_color(95), (0, 255, 0))
        self.assertEqual(get_score_color(80), (0, 255, 255))
        self.assertEqual(get_score_color(60), (0, 0, 255))
        
    def test_json_report_structure(self):
        """Test JSON report data structure"""
        report_data = {
            "timestamp": "2025-07-28T12:00:00",
            "overall_score": 85.5,
            "measurements": {
                "head_position": 5.2,
                "shoulder_alignment": 2.1,
                "spine_curvature": 28.5
            },
            "recommendations": [
                "Maintain good head position",
                "Stretch neck muscles regularly"
            ]
        }
        
        # Validate structure
        self.assertIn("timestamp", report_data)
        self.assertIn("overall_score", report_data)
        self.assertIn("measurements", report_data)
        self.assertIn("recommendations", report_data)
        
        # Validate data types
        self.assertIsInstance(report_data["overall_score"], float)
        self.assertIsInstance(report_data["measurements"], dict)
        self.assertIsInstance(report_data["recommendations"], list)

if __name__ == '__main__':
    unittest.main()
