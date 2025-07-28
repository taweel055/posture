#!/usr/bin/env python3
"""
Debug camera app to test what's happening
"""

import cv2
import time

def debug_camera():
    print("🎯 Starting camera debug...")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    print("✅ Camera opened successfully")
    
    # Wait a moment for camera to initialize
    time.sleep(2)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Check camera properties
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"📷 Camera: {width}x{height} @ {fps} FPS")
    
    print("📷 Camera properties set")
    print("Press 'q' to quit")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame_count += 1
            
            # Add frame counter to image
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Debug Camera', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🚪 Quit requested")
                break
                
            # Print status every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"📊 Frame {frame_count}: {fps:.1f} FPS")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"📊 Debug complete: {frame_count} frames, {fps:.1f} FPS")

if __name__ == "__main__":
    debug_camera()