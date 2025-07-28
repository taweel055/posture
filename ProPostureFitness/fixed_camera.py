#!/usr/bin/env python3
"""
Fixed camera app for macOS
"""

import cv2
import time

def test_camera():
    print("🎯 Testing camera with AVFoundation backend...")
    
    # Try AVFoundation backend specifically for macOS
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    
    if not cap.isOpened():
        print("❌ Cannot open camera with AVFoundation")
        # Fallback to default
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera at all")
            return
    
    print("✅ Camera opened successfully")
    
    frame_count = 0
    
    try:
        while frame_count < 100:  # Test for 100 frames
            ret, frame = cap.read()
            
            if not ret:
                print(f"❌ Failed to read frame {frame_count}")
                break
            
            frame_count += 1
            
            # Add frame counter
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Fixed Camera Test', frame)
            
            # Quick exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"📊 Processed {frame_count} frames")

if __name__ == "__main__":
    test_camera()