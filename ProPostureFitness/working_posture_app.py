#!/usr/bin/env python3
"""
Working Posture Analysis System for macOS
Fixed version that handles camera initialization properly
"""

import cv2
import numpy as np
import time
import threading
import sys
from datetime import datetime

# Try to import MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# Try to import PyTorch
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

class WorkingPostureApp:
    """Working posture analysis application"""
    
    def __init__(self):
        self.running = False
        self.camera = None
        self.frame_count = 0
        self.start_time = None
        
        # Initialize MediaPipe if available
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
                model_complexity=1  # Use simpler model for stability
            )
            print("✅ MediaPipe pose detection initialized")
        else:
            print("⚠️ MediaPipe not available")
            
        # Initialize PyTorch if available
        if PYTORCH_AVAILABLE:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("✅ PyTorch MPS acceleration available")
            else:
                self.device = torch.device("cpu")
                print("⚠️ PyTorch MPS not available, using CPU")
        else:
            print("⚠️ PyTorch not available")
    
    def init_camera(self):
        """Initialize camera with proper error handling"""
        print("📷 Initializing camera...")
        
        # Try different camera backends
        backends = [
            (cv2.CAP_AVFOUNDATION, "AVFoundation"),
            (cv2.CAP_ANY, "Default")
        ]
        
        for backend, name in backends:
            try:
                print(f"   Trying {name} backend...")
                if backend == cv2.CAP_ANY:
                    cap = cv2.VideoCapture(0)
                else:
                    cap = cv2.VideoCapture(0, backend)
                
                if cap.isOpened():
                    # Test if we can actually read a frame
                    print("   Camera opened, testing frame read...")
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        print(f"✅ Camera working with {name}: {test_frame.shape}")
                        self.camera = cap
                        return True
                    else:
                        print(f"   {name}: Camera opened but cannot read frames")
                        cap.release()
                else:
                    print(f"   {name}: Cannot open camera")
                    
            except Exception as e:
                print(f"   {name}: Error - {e}")
                
        print("❌ Could not initialize any camera")
        return False
    
    def analyze_posture(self, frame):
        """Analyze posture from frame"""
        if not MEDIAPIPE_AVAILABLE:
            return frame, None
            
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.pose.process(rgb_frame)
            
            # Draw landmarks if detected
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                
                # Simple posture score (placeholder)
                landmarks = results.pose_landmarks.landmark
                score = self.calculate_simple_score(landmarks)
                
                # Add score to frame
                cv2.putText(frame, f"Posture Score: {score:.1f}/100", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                return frame, score
                
        except Exception as e:
            print(f"⚠️ Posture analysis error: {e}")
            
        return frame, None
    
    def calculate_simple_score(self, landmarks):
        """Calculate a simple posture score"""
        try:
            # Get key landmarks
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # Calculate shoulder alignment
            shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
            alignment_score = max(0, 100 - (shoulder_diff * 1000))
            
            # Head position relative to shoulders
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            head_offset = abs(nose.x - shoulder_center_x)
            head_score = max(0, 100 - (head_offset * 200))
            
            # Combined score
            total_score = (alignment_score + head_score) / 2
            return min(100, max(0, total_score))
            
        except Exception:
            return 50.0  # Default neutral score
    
    def run(self):
        """Main application loop"""
        print("\n🎯 WORKING POSTURE ANALYSIS SYSTEM")
        print("=" * 50)
        print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Initialize camera
        if not self.init_camera():
            print("💡 Troubleshooting tips:")
            print("   1. Check camera permissions in System Preferences > Security & Privacy")
            print("   2. Close other applications using the camera")
            print("   3. Try running: sudo tccutil reset Camera")
            return
        
        print("\n🎥 Starting posture analysis...")
        print("Controls:")
        print("  SPACE - Take screenshot")
        print("  Q - Quit")
        print()
        
        self.running = True
        self.start_time = time.time()
        
        try:
            while self.running:
                if self.camera is None:
                    print("❌ Camera not initialized")
                    break
                    
                ret, frame = self.camera.read()
                
                if not ret:
                    print("❌ Lost camera connection")
                    break
                
                self.frame_count += 1
                
                # Analyze posture
                processed_frame, score = self.analyze_posture(frame)
                
                # Add frame info
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0
                
                cv2.putText(processed_frame, f"FPS: {fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(processed_frame, f"Frame: {self.frame_count}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Working Posture Analysis', processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("🚪 Quit requested")
                    break
                elif key == ord(' '):
                    # Save screenshot
                    filename = f"posture_screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"📸 Screenshot saved: {filename}")
                
                # Print status every 30 frames
                if self.frame_count % 30 == 0:
                    score_text = f", Score: {score:.1f}" if score else ""
                    print(f"📊 Frame {self.frame_count}: {fps:.1f} FPS{score_text}")
                    
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        except Exception as e:
            print(f"❌ Error during execution: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        
        if self.start_time:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            print(f"\n📊 Session Summary:")
            print(f"   Frames processed: {self.frame_count}")
            print(f"   Duration: {elapsed:.1f}s")
            print(f"   Average FPS: {fps:.1f}")
        
        print("🧹 Cleanup completed")

def main():
    """Main function"""
    app = WorkingPostureApp()
    app.run()

if __name__ == "__main__":
    main()
