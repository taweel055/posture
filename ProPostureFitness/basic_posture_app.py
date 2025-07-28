#!/usr/bin/env python3
"""
Basic Posture Analysis System - No MediaPipe to avoid hanging
"""

import cv2
import numpy as np
import time
from datetime import datetime

class BasicPostureApp:
    """Basic posture analysis without MediaPipe"""
    
    def __init__(self):
        self.running = False
        self.camera = None
        self.frame_count = 0
        self.start_time = None
        
        # Face detection for basic posture analysis
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            print("✅ Face detection initialized")
        except Exception as e:
            print(f"⚠️ Face detection not available: {e}")
            self.face_cascade = None
    
    def init_camera(self):
        """Initialize camera"""
        print("📷 Initializing camera...")
        
        try:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("❌ Cannot open camera")
                return False
            
            # Test frame read
            ret, test_frame = cap.read()
            if not ret:
                print("❌ Cannot read frames")
                cap.release()
                return False
            
            print(f"✅ Camera working: {test_frame.shape}")
            self.camera = cap
            return True
            
        except Exception as e:
            print(f"❌ Camera error: {e}")
            return False
    
    def analyze_posture(self, frame):
        """Basic posture analysis using face detection"""
        if self.face_cascade is None:
            return frame, None
        
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            score = 75.0  # Default score
            
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Simple posture analysis based on face position
                frame_center_x = frame.shape[1] // 2
                face_center_x = x + w // 2
                
                # Calculate head tilt score
                offset = abs(face_center_x - frame_center_x)
                max_offset = frame.shape[1] // 4
                tilt_score = max(0, 100 - (offset / max_offset * 50))
                
                # Head height analysis
                head_height_ratio = y / frame.shape[0]
                height_score = 100 if 0.1 < head_height_ratio < 0.4 else 50
                
                score = (tilt_score + height_score) / 2
                
                # Add posture feedback
                if score > 80:
                    status = "GOOD"
                    color = (0, 255, 0)
                elif score > 60:
                    status = "FAIR"
                    color = (0, 255, 255)
                else:
                    status = "POOR"
                    color = (0, 0, 255)
                
                cv2.putText(frame, status, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
            return frame, score
            
        except Exception as e:
            print(f"⚠️ Analysis error: {e}")
            return frame, None
    
    def run(self):
        """Main application loop"""
        print("\n🎯 BASIC POSTURE ANALYSIS SYSTEM")
        print("=" * 50)
        print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Initialize camera
        if not self.init_camera():
            print("💡 Try granting camera permissions in System Preferences")
            return
        
        print("\n🎥 Starting basic posture analysis...")
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
                
                if score is not None:
                    cv2.putText(processed_frame, f"Posture Score: {score:.1f}/100", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(processed_frame, f"Frame: {self.frame_count}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Basic Posture Analysis', processed_frame)
                
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
                
                # Print status every 60 frames
                if self.frame_count % 60 == 0:
                    score_text = f", Score: {score:.1f}" if score else ""
                    print(f"📊 Frame {self.frame_count}: {fps:.1f} FPS{score_text}")
                    
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
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
    app = BasicPostureApp()
    app.run()

if __name__ == "__main__":
    main()
