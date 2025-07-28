#!/usr/bin/env python3
"""
Simple camera test without MediaPipe to isolate the issue
"""

import cv2
import sys

def simple_test():
    print("🎯 Simple Camera Test")
    print("=" * 30)
    
    try:
        print("📷 Opening camera...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Camera not available")
            return False
        
        print("✅ Camera opened")
        
        # Try to read one frame
        print("📸 Reading test frame...")
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Cannot read frame")
            cap.release()
            return False
        
        print(f"✅ Frame read successfully: {frame.shape}")
        
        # Show frame for 3 seconds
        cv2.imshow('Simple Test', frame)
        cv2.waitKey(3000)
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = simple_test()
    sys.exit(0 if success else 1)