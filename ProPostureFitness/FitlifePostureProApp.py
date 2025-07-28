#!/usr/bin/env python3
"""
FitlifePostureProApp - Professional Posture Analysis System
Advanced posture monitoring and fitness assessment application
"""

import cv2
import numpy as np
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
import argparse
import sys

from config_module import config

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

class AnalysisMode(Enum):
    """Analysis mode selection for FitlifePostureProApp"""
    BASIC = "basic"
    ADVANCED = "advanced"
    GPU_ACCELERATED = "gpu_accelerated"

class PostureAnalyzer(ABC):
    """Base class for all posture analyzers in FitlifePostureProApp"""
    
    def __init__(self, mode: AnalysisMode):
        self.mode = mode
        self.running = False
        self.frame_count = 0
        self.start_time = None
        
    @abstractmethod
    def analyze_posture(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        """Analyze posture from frame - must be implemented by subclasses"""
        pass
        
    @abstractmethod
    def get_required_dependencies(self) -> List[str]:
        """Return list of required dependencies for this analyzer"""
        pass

class BasicPostureAnalyzer(PostureAnalyzer):
    """Basic posture analysis using face detection only"""
    
    def __init__(self):
        super().__init__(AnalysisMode.BASIC)
        
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            print("✅ Face detection initialized")
        except Exception as e:
            print(f"⚠️ Face detection not available: {e}")
            self.face_cascade = None
    
    def get_required_dependencies(self) -> List[str]:
        return ["opencv-python"]
    
    def analyze_posture(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        """Basic posture analysis using face detection"""
        if self.face_cascade is None:
            return frame, None
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            score = 75.0
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                frame_center_x = frame.shape[1] // 2
                face_center_x = x + w // 2
                
                offset = abs(face_center_x - frame_center_x)
                max_offset = frame.shape[1] // 4
                tilt_score = max(0, 100 - (offset / max_offset * 50))
                
                head_height_ratio = y / frame.shape[0]
                height_score = 100 if 0.1 < head_height_ratio < 0.4 else 50
                
                score = (tilt_score + height_score) / 2
                
                if score > 80:
                    status = "EXCELLENT"
                    color = (0, 255, 0)
                elif score > 60:
                    status = "GOOD"
                    color = (0, 255, 255)
                else:
                    status = "NEEDS IMPROVEMENT"
                    color = (0, 0, 255)
                
                cv2.putText(frame, status, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                break
                
            return frame, score
            
        except Exception as e:
            print(f"⚠️ Analysis error: {e}")
            return frame, None

class AdvancedPostureAnalyzer(PostureAnalyzer):
    """Advanced posture analysis using MediaPipe pose detection"""
    
    def __init__(self):
        super().__init__(AnalysisMode.ADVANCED)
        
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_pose = mp.solutions.pose
                self.mp_drawing = mp.solutions.drawing_utils
                self.pose = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5
                )
                print("✅ MediaPipe pose detection initialized")
            except Exception as e:
                print(f"⚠️ MediaPipe initialization failed: {e}")
                self.pose = None
        else:
            print("⚠️ MediaPipe not available - falling back to basic analysis")
            self.pose = None
    
    def get_required_dependencies(self) -> List[str]:
        return ["opencv-python", "mediapipe"]
    
    def analyze_posture(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        """Advanced posture analysis using MediaPipe"""
        if not self.pose:
            return frame, None
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                )
                
                score = self._calculate_posture_score(results.pose_landmarks)
                
                if score > 85:
                    status = "EXCELLENT POSTURE"
                    color = (0, 255, 0)
                elif score > 70:
                    status = "GOOD POSTURE"
                    color = (0, 255, 255)
                elif score > 50:
                    status = "FAIR POSTURE"
                    color = (255, 255, 0)
                else:
                    status = "POOR POSTURE"
                    color = (0, 0, 255)
                
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"Score: {score:.1f}/100", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                return frame, score
            
            return frame, None
            
        except Exception as e:
            print(f"⚠️ MediaPipe analysis error: {e}")
            return frame, None
    
    def _calculate_posture_score(self, landmarks) -> float:
        """Calculate comprehensive posture score"""
        try:
            lm = landmarks.landmark
            
            left_shoulder = [lm[11].x, lm[11].y]
            right_shoulder = [lm[12].x, lm[12].y]
            nose = [lm[0].x, lm[0].y]
            
            shoulder_center = [(left_shoulder[0] + right_shoulder[0]) / 2,
                             (left_shoulder[1] + right_shoulder[1]) / 2]
            
            head_alignment = abs(nose[0] - shoulder_center[0])
            shoulder_level = abs(left_shoulder[1] - right_shoulder[1])
            
            head_score = max(0, 100 - (head_alignment * 1000))
            shoulder_score = max(0, 100 - (shoulder_level * 1000))
            
            overall_score = (head_score + shoulder_score) / 2
            return min(100, max(0, overall_score))
            
        except Exception as e:
            print(f"⚠️ Score calculation error: {e}")
            return 50.0

class GPUAcceleratedPostureAnalyzer(PostureAnalyzer):
    """GPU-accelerated posture analysis for maximum performance"""
    
    def __init__(self):
        super().__init__(AnalysisMode.GPU_ACCELERATED)
        
        self.gpu_available = False
        if PYTORCH_AVAILABLE:
            try:
                import torch
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.gpu_available = torch.cuda.is_available()
                print(f"✅ GPU acceleration: {'Enabled' if self.gpu_available else 'CPU fallback'}")
            except Exception as e:
                print(f"⚠️ GPU initialization failed: {e}")
                self.device = 'cpu'
        
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_pose = mp.solutions.pose
                self.mp_drawing = mp.solutions.drawing_utils
                self.pose = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=2,
                    enable_segmentation=False,
                    min_detection_confidence=0.8,
                    min_tracking_confidence=0.7
                )
                print("✅ High-performance MediaPipe initialized")
            except Exception as e:
                print(f"⚠️ MediaPipe GPU setup failed: {e}")
                self.pose = None
        else:
            self.pose = None
    
    def get_required_dependencies(self) -> List[str]:
        return ["opencv-python", "mediapipe", "torch"]
    
    def analyze_posture(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        """GPU-accelerated posture analysis"""
        if not self.pose:
            return frame, None
        
        try:
            start_time = time.time()
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            processing_time = (time.time() - start_time) * 1000
            
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                )
                
                score = self._calculate_advanced_score(results.pose_landmarks)
                
                cv2.putText(frame, f"GPU Mode - Score: {score:.1f}/100", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Processing: {processing_time:.1f}ms", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if self.gpu_available:
                    cv2.putText(frame, "GPU ACCELERATED", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                return frame, score
            
            return frame, None
            
        except Exception as e:
            print(f"⚠️ GPU analysis error: {e}")
            return frame, None
    
    def _calculate_advanced_score(self, landmarks) -> float:
        """Advanced posture scoring with multiple metrics"""
        try:
            lm = landmarks.landmark
            
            scores = []
            
            left_shoulder = [lm[11].x, lm[11].y, lm[11].z]
            right_shoulder = [lm[12].x, lm[12].y, lm[12].z]
            nose = [lm[0].x, lm[0].y, lm[0].z]
            left_ear = [lm[7].x, lm[7].y, lm[7].z]
            right_ear = [lm[8].x, lm[8].y, lm[8].z]
            
            shoulder_center = [(left_shoulder[0] + right_shoulder[0]) / 2,
                             (left_shoulder[1] + right_shoulder[1]) / 2,
                             (left_shoulder[2] + right_shoulder[2]) / 2]
            
            head_forward = abs(nose[2] - shoulder_center[2])
            head_alignment = abs(nose[0] - shoulder_center[0])
            shoulder_level = abs(left_shoulder[1] - right_shoulder[1])
            ear_alignment = abs(left_ear[1] - right_ear[1])
            
            scores.append(max(0, 100 - (head_forward * 500)))
            scores.append(max(0, 100 - (head_alignment * 1000)))
            scores.append(max(0, 100 - (shoulder_level * 1000)))
            scores.append(max(0, 100 - (ear_alignment * 1000)))
            
            overall_score = sum(scores) / len(scores)
            return min(100, max(0, overall_score))
            
        except Exception as e:
            print(f"⚠️ Advanced score calculation error: {e}")
            return 50.0

class CameraManager:
    """Professional camera management for FitlifePostureProApp"""
    
    def __init__(self):
        self.camera = None
        self.camera_settings = config.settings.get('settings', {})
        
    def init_camera(self) -> bool:
        """Initialize camera with professional-grade error handling"""
        print("📷 Initializing camera for FitlifePostureProApp...")
        
        backends = [
            (cv2.CAP_AVFOUNDATION, "AVFoundation"),
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_V4L2, "Video4Linux2"),
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
                    print("   Camera opened, testing frame read...")
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        print(f"✅ Camera working with {name}: {test_frame.shape}")
                        
                        self._apply_camera_settings(cap)
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
    
    def _apply_camera_settings(self, cap):
        """Apply professional camera settings"""
        try:
            resolution = self.camera_settings.get('camera_resolution', '1280x720')
            width, height = map(int, resolution.split('x'))
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, self.camera_settings.get('processing_fps', 30))
            
            print(f"   Applied settings: {resolution} @ {self.camera_settings.get('processing_fps', 30)}fps")
            
        except Exception as e:
            print(f"   Warning: Could not apply camera settings: {e}")
    
    def release(self):
        """Release camera resources"""
        if self.camera:
            self.camera.release()
            self.camera = None

class FitlifePostureProApp:
    """Main FitlifePostureProApp application class"""
    
    def __init__(self, mode: Optional[AnalysisMode] = None):
        print("🏃‍♂️ FITLIFE POSTURE PRO APP")
        print("=" * 60)
        print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Professional Posture Analysis & Fitness Assessment")
        print()
        
        self.mode = mode or self._determine_optimal_mode()
        print(f"📊 Analysis Mode: {self.mode.value.upper()}")
        
        self.camera_manager = CameraManager()
        self.analyzer = self._create_analyzer()
        
        self._check_dependencies()
        
        print("✅ FitlifePostureProApp initialized")
        print()
    
    def _determine_optimal_mode(self) -> AnalysisMode:
        """Determine optimal analysis mode based on system capabilities"""
        settings = config.settings.get('settings', {})
        gpu_enabled = settings.get('gpu_acceleration', True)
        
        if gpu_enabled and PYTORCH_AVAILABLE and MEDIAPIPE_AVAILABLE:
            return AnalysisMode.GPU_ACCELERATED
        elif MEDIAPIPE_AVAILABLE:
            return AnalysisMode.ADVANCED
        else:
            print("⚠️ Falling back to basic mode - MediaPipe not available")
            return AnalysisMode.BASIC
    
    def _create_analyzer(self) -> PostureAnalyzer:
        """Create appropriate analyzer based on mode"""
        if self.mode == AnalysisMode.BASIC:
            return BasicPostureAnalyzer()
        elif self.mode == AnalysisMode.ADVANCED:
            return AdvancedPostureAnalyzer()
        elif self.mode == AnalysisMode.GPU_ACCELERATED:
            return GPUAcceleratedPostureAnalyzer()
        else:
            raise ValueError(f"Unknown analysis mode: {self.mode}")
    
    def _check_dependencies(self):
        """Check and report dependency status"""
        required_deps = self.analyzer.get_required_dependencies()
        print(f"📋 Required dependencies for {self.mode.value} mode:")
        for dep in required_deps:
            print(f"   ✅ {dep}")
        print()
    
    def run(self):
        """Run the main application loop"""
        if not self.camera_manager.init_camera():
            print("💡 Please check camera permissions and try again")
            return
        
        print("🎥 Starting FitlifePostureProApp analysis...")
        print("Controls:")
        print("  SPACE - Take screenshot")
        print("  S - Save current analysis")
        print("  R - Reset session")
        print("  H - Show help")
        print("  Q - Quit")
        print()
        
        self.analyzer.running = True
        self.analyzer.start_time = time.time()
        
        try:
            while self.analyzer.running:
                if self.camera_manager.camera is None:
                    print("❌ Camera not available")
                    break
                    
                ret, frame = self.camera_manager.camera.read()
                
                if not ret:
                    print("❌ Lost camera connection")
                    break
                
                self.analyzer.frame_count += 1
                
                processed_frame, score = self.analyzer.analyze_posture(frame)
                
                elapsed = time.time() - self.analyzer.start_time
                fps = self.analyzer.frame_count / elapsed if elapsed > 0 else 0
                
                cv2.putText(processed_frame, f"FitlifePostureProApp v1.0", 
                           (10, processed_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(processed_frame, f"FPS: {fps:.1f} | Frame: {self.analyzer.frame_count}", 
                           (10, processed_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('FitlifePostureProApp - Professional Posture Analysis', processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("🚪 Quit requested")
                    break
                elif key == ord(' '):
                    filename = f"fitlife_posture_screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"📸 Screenshot saved: {filename}")
                elif key == ord('s'):
                    print(f"💾 Analysis saved - Score: {score:.1f}" if score else "💾 Analysis saved")
                elif key == ord('r'):
                    print("🔄 Session reset")
                    self.analyzer.frame_count = 0
                    self.analyzer.start_time = time.time()
                elif key == ord('h'):
                    self._show_help()
                
                if self.analyzer.frame_count % 60 == 0:
                    score_text = f", Score: {score:.1f}" if score else ""
                    print(f"📊 Frame {self.analyzer.frame_count}: {fps:.1f} FPS{score_text}")
                    
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self._cleanup()
    
    def _show_help(self):
        """Display help information"""
        print("\n" + "="*50)
        print("FITLIFE POSTURE PRO APP - HELP")
        print("="*50)
        print("SPACE - Take screenshot")
        print("S     - Save current analysis")
        print("R     - Reset session")
        print("H     - Show this help")
        print("Q     - Quit application")
        print("="*50)
        print()
    
    def _cleanup(self):
        """Clean up resources"""
        self.analyzer.running = False
        self.camera_manager.release()
        cv2.destroyAllWindows()
        
        if self.analyzer.start_time:
            elapsed = time.time() - self.analyzer.start_time
            fps = self.analyzer.frame_count / elapsed if elapsed > 0 else 0
            print(f"\n📊 Session Summary:")
            print(f"   Frames processed: {self.analyzer.frame_count}")
            print(f"   Duration: {elapsed:.1f}s")
            print(f"   Average FPS: {fps:.1f}")
        
        print("🧹 FitlifePostureProApp session completed")

def main():
    """Main entry point for FitlifePostureProApp"""
    parser = argparse.ArgumentParser(description='FitlifePostureProApp - Professional Posture Analysis')
    parser.add_argument('--mode', choices=['basic', 'advanced', 'gpu'], 
                       help='Force specific analysis mode')
    parser.add_argument('--config', help='Path to custom config file')
    
    args = parser.parse_args()
    
    mode = None
    if args.mode:
        mode = AnalysisMode(args.mode)
    
    try:
        app = FitlifePostureProApp(mode=mode)
        app.run()
        
    except KeyboardInterrupt:
        print("\n🛑 FitlifePostureProApp stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
