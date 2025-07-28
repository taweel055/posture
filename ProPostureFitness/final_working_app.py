#!/usr/bin/env python3
"""
Final Working Posture Analysis System
Simplified version that works reliably on macOS
"""

import cv2
import numpy as np
import time
import sys
from datetime import datetime

class FinalPostureApp:
    """Final working posture analysis application"""
    
    def __init__(self):
        print("🎯 PROPOSTUREFITNESS v5.0 - WORKING EDITION")
        print("=" * 60)
        print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("📱 Optimized for macOS with Apple Silicon")
        print()
        
        self.running = False
        self.camera = None
        self.frame_count = 0
        self.start_time = None
        
        # Initialize face detection for basic posture analysis
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            print("✅ Face detection initialized")
        except Exception as e:
            print(f"⚠️ Face detection not available: {e}")
            self.face_cascade = None
            
        # Initialize body detection
        try:
            self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
            print("✅ Body detection initialized")
        except Exception as e:
            print("⚠️ Body detection not available")
            self.body_cascade = None
    
    def init_camera(self):
        """Initialize camera with robust error handling"""
        print("\n📷 Initializing camera system...")
        
        try:
            # Simple camera initialization
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("❌ Cannot open camera")
                return False
            
            print("✅ Camera opened successfully")
            
            # Set optimal properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Verify we can read frames
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                print("❌ Cannot read frames from camera")
                cap.release()
                return False
            
            print(f"✅ Camera working: {test_frame.shape}")
            print(f"   Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            print(f"   FPS: {cap.get(cv2.CAP_PROP_FPS)}")
            
            self.camera = cap
            return True
            
        except Exception as e:
            print(f"❌ Camera initialization error: {e}")
            return False
    
    def analyze_posture(self, frame):
        """Analyze posture using computer vision"""
        analysis_frame = frame.copy()
        score = 75.0  # Default neutral score
        
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Face detection for head position
            if self.face_cascade is not None:
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
                
                for (x, y, w, h) in faces:
                    # Draw face rectangle
                    cv2.rectangle(analysis_frame, (x, y), (x+w, y+h), (255, 100, 0), 2)
                    
                    # Analyze head position
                    frame_center_x = frame.shape[1] // 2
                    face_center_x = x + w // 2
                    
                    # Head alignment score
                    offset = abs(face_center_x - frame_center_x)
                    max_offset = frame.shape[1] // 3
                    alignment_score = max(0, 100 - (offset / max_offset * 40))
                    
                    # Head height analysis
                    head_ratio = y / frame.shape[0]
                    if 0.05 < head_ratio < 0.4:
                        height_score = 100
                    elif 0.4 <= head_ratio < 0.6:
                        height_score = 80
                    else:
                        height_score = 50
                    
                    # Face size analysis (distance from camera)
                    face_area = w * h
                    frame_area = frame.shape[0] * frame.shape[1]
                    face_ratio = face_area / frame_area
                    
                    if 0.02 < face_ratio < 0.15:
                        distance_score = 100
                    else:
                        distance_score = 70
                    
                    # Combined score
                    score = (alignment_score + height_score + distance_score) / 3
                    
                    # Posture feedback
                    if score >= 85:
                        status = "EXCELLENT"
                        color = (0, 255, 0)
                    elif score >= 70:
                        status = "GOOD"
                        color = (0, 200, 100)
                    elif score >= 55:
                        status = "FAIR"
                        color = (0, 255, 255)
                    else:
                        status = "POOR"
                        color = (0, 100, 255)
                    
                    # Add status text
                    cv2.putText(analysis_frame, status, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    # Add center line for reference
                    cv2.line(analysis_frame, (frame_center_x, 0), 
                            (frame_center_x, frame.shape[0]), (200, 200, 200), 1)
                    
                    break  # Only analyze first face
            
            return analysis_frame, score
            
        except Exception as e:
            print(f"⚠️ Analysis error: {e}")
            return frame, score
    
    def run(self):
        """Main application loop"""
        print("\n🎥 STARTING POSTURE ANALYSIS")
        print("-" * 40)
        
        # Initialize camera
        if not self.init_camera():
            print("\n💡 TROUBLESHOOTING TIPS:")
            print("   1. Check camera permissions: System Preferences > Security & Privacy > Camera")
            print("   2. Close other apps using camera (Zoom, Teams, etc.)")
            print("   3. Restart Terminal and try again")
            print("   4. Try: sudo tccutil reset Camera")
            return False
        
        print("\n🎮 CONTROLS:")
        print("   SPACE - Take screenshot")
        print("   Q - Quit application")
        print("   ESC - Emergency exit")
        print()
        print("🔄 Starting real-time analysis...")
        
        self.running = True
        self.start_time = time.time()
        last_status_time = time.time()
        
        try:
            while self.running:
                if self.camera is None:
                    print("❌ Camera not initialized")
                    break
                    
                ret, frame = self.camera.read()
                
                if not ret:
                    print("❌ Camera disconnected")
                    break
                
                self.frame_count += 1
                
                # Analyze posture
                processed_frame, score = self.analyze_posture(frame)
                
                # Add performance info
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0
                
                # Add overlay information
                cv2.putText(processed_frame, f"FPS: {fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(processed_frame, f"Posture Score: {score:.0f}/100", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(processed_frame, f"Frames: {self.frame_count}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add timestamp
                timestamp = datetime.now().strftime("%H:%M:%S")
                cv2.putText(processed_frame, timestamp, 
                           (processed_frame.shape[1] - 120, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                # Display frame
                cv2.imshow('ProPostureFitness v5.0 - Real-time Analysis', processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Q or ESC
                    print("🚪 Quit requested")
                    break
                elif key == ord(' '):
                    # Save screenshot
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"posture_analysis_{timestamp_str}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"📸 Screenshot saved: {filename} (Score: {score:.0f})")
                
                # Print status every 120 frames (about every 4 seconds at 30fps)
                if time.time() - last_status_time > 5:
                    print(f"📊 Running: {fps:.1f} FPS | Score: {score:.0f}/100 | Frames: {self.frame_count}")
                    last_status_time = time.time()
                    
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user (Ctrl+C)")
        except Exception as e:
            print(f"❌ Runtime error: {e}")
        finally:
            self.cleanup()
            return True
    
    def cleanup(self):
        """Clean up resources"""
        print("\n🧹 Cleaning up...")
        
        self.running = False
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        
        if self.start_time:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            
            print("📊 SESSION SUMMARY:")
            print(f"   Duration: {elapsed:.1f} seconds")
            print(f"   Frames processed: {self.frame_count}")
            print(f"   Average FPS: {fps:.1f}")
            print(f"   Performance: {'Excellent' if fps > 25 else 'Good' if fps > 15 else 'Fair'}")
        
        print("✅ ProPostureFitness session completed")

def main():
    """Main entry point"""
    try:
        app = FinalPostureApp()
        success = app.run()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Application failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
