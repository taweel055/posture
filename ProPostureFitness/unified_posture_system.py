#!/usr/bin/env python3
"""
Unified Posture Analysis System
Consolidates all posture analysis modes into a single, configurable system
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
    """Analysis mode selection"""
    BASIC = "basic"
    ADVANCED = "advanced"
    GPU_ACCELERATED = "gpu_accelerated"

class PostureAnalyzer(ABC):
    """Base class for all posture analyzers"""
    
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
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
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
                    status = "GOOD"
                    color = (0, 255, 0)
                elif score > 60:
                    status = "FAIR"
                    color = (0, 255, 255)
                else:
                    status = "POOR"
                    color = (0, 0, 255)
                
                cv2.putText(frame, status, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                break
                
            return frame, score
            
        except Exception as e:
            print(f"⚠️ Analysis error: {e}")
            return frame, None

class AdvancedPostureAnalyzer(PostureAnalyzer):
    """Advanced posture analysis using MediaPipe or enhanced face detection"""
    
    def __init__(self):
        super().__init__(AnalysisMode.ADVANCED)
        
        self.scoring_algorithm = config.settings.get('settings', {}).get('advanced_mode', {}).get('scoring_algorithm', 'mediapipe')
        
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            
            advanced_settings = config.settings.get('settings', {}).get('advanced_mode', {})
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=advanced_settings.get('detection_confidence', 0.7),
                min_tracking_confidence=advanced_settings.get('tracking_confidence', 0.5),
                model_complexity=advanced_settings.get('mediapipe_complexity', 1)
            )
            print("✅ MediaPipe pose detection initialized")
        else:
            print("⚠️ MediaPipe not available")
            self.mp_pose = None
            self.pose = None
            
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
            print("✅ Enhanced face and body detection initialized")
        except Exception as e:
            print(f"⚠️ Enhanced detection not available: {e}")
            self.face_cascade = None
            self.body_cascade = None
    
    def get_required_dependencies(self) -> List[str]:
        return ["opencv-python", "mediapipe"]
    
    def analyze_posture(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        """Advanced posture analysis"""
        if MEDIAPIPE_AVAILABLE and self.pose and self.scoring_algorithm == 'mediapipe':
            return self._analyze_with_mediapipe(frame)
        else:
            return self._analyze_with_enhanced_face_detection(frame)
    
    def _analyze_with_mediapipe(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        """MediaPipe-based analysis"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                
                landmarks = results.pose_landmarks.landmark
                score = self._calculate_mediapipe_score(landmarks)
                
                cv2.putText(frame, f"Posture Score: {score:.1f}/100", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                return frame, score
                
        except Exception as e:
            print(f"⚠️ MediaPipe analysis error: {e}")
            
        return frame, None
    
    def _calculate_mediapipe_score(self, landmarks) -> float:
        """Calculate posture score from MediaPipe landmarks"""
        try:
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
            alignment_score = max(0, 100 - (shoulder_diff * 1000))
            
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            head_offset = abs(nose.x - shoulder_center_x)
            head_score = max(0, 100 - (head_offset * 200))
            
            total_score = (alignment_score + head_score) / 2
            return min(100, max(0, total_score))
            
        except Exception:
            return 50.0
    
    def _analyze_with_enhanced_face_detection(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        """Enhanced face detection analysis"""
        if self.face_cascade is None:
            return frame, None
            
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
            
            score = 75.0
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 100, 0), 2)
                
                frame_center_x = frame.shape[1] // 2
                face_center_x = x + w // 2
                
                offset = abs(face_center_x - frame_center_x)
                max_offset = frame.shape[1] // 3
                alignment_score = max(0, 100 - (offset / max_offset * 40))
                
                head_ratio = y / frame.shape[0]
                if 0.05 < head_ratio < 0.4:
                    height_score = 100
                elif 0.4 <= head_ratio < 0.6:
                    height_score = 80
                else:
                    height_score = 50
                
                face_area = w * h
                frame_area = frame.shape[0] * frame.shape[1]
                face_ratio = face_area / frame_area
                
                if 0.02 < face_ratio < 0.15:
                    distance_score = 100
                else:
                    distance_score = 70
                
                score = (alignment_score + height_score + distance_score) / 3
                
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
                
                cv2.putText(frame, status, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                cv2.line(frame, (frame_center_x, 0), 
                        (frame_center_x, frame.shape[0]), (200, 200, 200), 1)
                
                break
            
            return frame, score
            
        except Exception as e:
            print(f"⚠️ Enhanced analysis error: {e}")
            return frame, score

class GPUAcceleratedPostureAnalyzer(PostureAnalyzer):
    """GPU-accelerated posture analysis using PyTorch and MediaPipe"""
    
    def __init__(self):
        super().__init__(AnalysisMode.GPU_ACCELERATED)
        
        from gpu_accelerated_posture_system import (
            GPUDeviceManager, GPUAcceleratedMediaPipe, 
            GPUAcceleratedImageProcessing, GPUAcceleratedPostureCalculator,
            GPUPerformanceMetrics
        )
        
        self.device_manager = GPUDeviceManager()
        self.mediapipe_gpu = GPUAcceleratedMediaPipe(self.device_manager)
        self.image_processor = GPUAcceleratedImageProcessing(self.device_manager)
        self.posture_calculator = GPUAcceleratedPostureCalculator(self.device_manager)
        
        self.performance_metrics = GPUPerformanceMetrics()
        self.frame_times = []
        self.max_frame_history = 30
        
        gpu_settings = config.settings.get('settings', {}).get('gpu_mode', {})
        self.show_performance = gpu_settings.get('performance_overlay', True)
        
        print("✅ GPU-accelerated system initialized")
        print(f"   Device: {self.device_manager.device}")
        print(f"   GPU Available: {self.device_manager.gpu_available}")
    
    def get_required_dependencies(self) -> List[str]:
        return ["opencv-python", "mediapipe", "torch", "numpy"]
    
    def analyze_posture(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        """GPU-accelerated posture analysis"""
        try:
            angles, annotated_frame, metrics = self._process_frame_complete(frame)
            
            if self.show_performance:
                self._add_performance_overlay(annotated_frame, metrics, angles)
            
            overall_score = self._calculate_overall_score(angles)
            return annotated_frame, overall_score
            
        except Exception as e:
            print(f"⚠️ GPU analysis error: {e}")
            return frame, None
    
    def _process_frame_complete(self, frame: np.ndarray):
        """Complete GPU-accelerated frame processing pipeline"""
        total_start_time = time.time()
        
        preprocessed_frame, preprocess_time = self.image_processor.preprocess_frame_gpu(frame)
        pose_results, inference_time = self.mediapipe_gpu.process_frame_gpu(preprocessed_frame)
        
        if (pose_results and 
            hasattr(pose_results, 'pose_landmarks') and 
            pose_results.pose_landmarks and
            hasattr(pose_results.pose_landmarks, 'landmark')):
            landmarks = [
                (lm.x, lm.y, lm.z) for lm in pose_results.pose_landmarks.landmark
            ]
            angles, calc_time = self.posture_calculator.calculate_angles_gpu(landmarks)
            
            annotated_frame = frame.copy()
            if (self.mediapipe_gpu.mp_drawing and 
                self.mediapipe_gpu.mp_pose and 
                hasattr(self.mediapipe_gpu.mp_pose, 'POSE_CONNECTIONS')):
                self.mediapipe_gpu.mp_drawing.draw_landmarks(
                    annotated_frame,
                    pose_results.pose_landmarks,
                    self.mediapipe_gpu.mp_pose.POSE_CONNECTIONS,
                    self.mediapipe_gpu.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mediapipe_gpu.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                )
        else:
            angles = {}
            calc_time = 0.0
            annotated_frame = frame
        
        total_time = (time.time() - total_start_time) * 1000
        self._update_performance_metrics(preprocess_time, inference_time, calc_time, total_time)
        
        return angles, annotated_frame, self.performance_metrics
    
    def _update_performance_metrics(self, preprocess_time: float, inference_time: float, 
                                   calc_time: float, total_time: float):
        """Update performance metrics"""
        current_time = time.time()
        self.frame_times.append(current_time)
        
        if len(self.frame_times) > self.max_frame_history:
            self.frame_times.pop(0)
        
        if len(self.frame_times) > 1:
            time_span = self.frame_times[-1] - self.frame_times[0]
            fps = (len(self.frame_times) - 1) / time_span if time_span > 0 else 0
        else:
            fps = 0
        
        self.performance_metrics.fps = fps
        self.performance_metrics.preprocessing_time_ms = preprocess_time
        self.performance_metrics.inference_time_ms = inference_time
        self.performance_metrics.postprocessing_time_ms = calc_time
        self.performance_metrics.total_time_ms = total_time
        
        if self.device_manager.gpu_available:
            memory_info = self.device_manager.get_gpu_memory_info()
            self.performance_metrics.gpu_memory_used = memory_info['allocated']
    
    def _add_performance_overlay(self, frame: np.ndarray, metrics, angles: Dict[str, float]):
        """Add performance information overlay to frame"""
        y_offset = 30
        cv2.putText(frame, f"FPS: {metrics.fps:.1f}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        y_offset += 30
        cv2.putText(frame, f"GPU: {self.device_manager.device}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        y_offset += 30
        cv2.putText(frame, f"Total: {metrics.total_time_ms:.1f}ms", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if self.device_manager.gpu_available:
            y_offset += 30
            cv2.putText(frame, f"GPU Mem: {metrics.gpu_memory_used:.1f}GB", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        y_offset += 50
        cv2.putText(frame, "Posture Measurements:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for param, value in angles.items():
            y_offset += 25
            display_name = param.replace('_', ' ').title()
            cv2.putText(frame, f"{display_name}: {value:.1f}°", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _calculate_overall_score(self, angles: Dict[str, float]) -> float:
        """Calculate overall posture score from angles"""
        if not angles:
            return 75.0
        
        try:
            head_tilt = angles.get('head_tilt', 0)
            shoulder_alignment = angles.get('shoulder_alignment', 0)
            forward_head = angles.get('forward_head_posture', 0)
            
            head_score = max(0, 100 - head_tilt * 2)
            shoulder_score = max(0, 100 - shoulder_alignment * 2)
            forward_score = max(0, 100 - forward_head)
            
            overall_score = (head_score + shoulder_score + forward_score) / 3
            return min(100, max(0, overall_score))
            
        except Exception:
            return 75.0

class CameraManager:
    """Unified camera management for all analysis modes"""
    
    def __init__(self):
        self.camera = None
        self.camera_settings = config.settings.get('settings', {})
        
    def init_camera(self) -> bool:
        """Initialize camera with robust error handling"""
        print("📷 Initializing camera...")
        
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
        """Apply camera settings from configuration"""
        try:
            resolution = self.camera_settings.get('camera_resolution', '1280x720')
            width, height = map(int, resolution.split('x'))
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, self.camera_settings.get('processing_fps', 30))
            
            print(f"   Applied settings: {width}x{height} @ {self.camera_settings.get('processing_fps', 30)}fps")
            
        except Exception as e:
            print(f"   Warning: Could not apply camera settings: {e}")
    
    def release(self):
        """Release camera resources"""
        if self.camera:
            self.camera.release()
            self.camera = None

class PostureAnalysisSystem:
    """Main unified posture analysis system"""
    
    def __init__(self, mode: Optional[AnalysisMode] = None):
        print("🎯 UNIFIED POSTURE ANALYSIS SYSTEM")
        print("=" * 60)
        print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.mode = mode or self._determine_optimal_mode()
        print(f"📊 Analysis Mode: {self.mode.value.upper()}")
        
        self.camera_manager = CameraManager()
        self.analyzer = self._create_analyzer()
        
        self._check_dependencies()
        
        print("✅ Unified system initialized")
        print()
    
    def _determine_optimal_mode(self) -> AnalysisMode:
        """Determine optimal analysis mode based on configuration and dependencies"""
        settings = config.settings.get('settings', {})
        analysis_mode = settings.get('analysis_mode', 'auto')
        
        if analysis_mode != 'auto':
            try:
                return AnalysisMode(analysis_mode)
            except ValueError:
                print(f"⚠️ Invalid analysis mode '{analysis_mode}', using auto-detection")
        
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
        """Check if required dependencies are available"""
        required_deps = self.analyzer.get_required_dependencies()
        missing_deps = []
        
        for dep in required_deps:
            if dep == "mediapipe" and not MEDIAPIPE_AVAILABLE:
                missing_deps.append(dep)
            elif dep == "torch" and not PYTORCH_AVAILABLE:
                missing_deps.append(dep)
        
        if missing_deps:
            print(f"⚠️ Missing dependencies: {', '.join(missing_deps)}")
            if self.mode != AnalysisMode.BASIC:
                print("   System will attempt graceful degradation")
    
    def run(self):
        """Main application loop"""
        print(f"\n🎥 STARTING {self.mode.value.upper()} POSTURE ANALYSIS")
        print("-" * 55)
        
        if not self.camera_manager.init_camera():
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
        if self.mode == AnalysisMode.GPU_ACCELERATED:
            print("   P - Toggle performance overlay")
        print()
        print("🔄 Starting real-time analysis...")
        
        self.analyzer.running = True
        self.analyzer.start_time = time.time()
        last_status_time = time.time()
        
        try:
            while self.analyzer.running:
                if self.camera_manager.camera is None:
                    print("❌ Camera not initialized")
                    break
                    
                ret, frame = self.camera_manager.camera.read()
                
                if not ret:
                    print("❌ Camera disconnected")
                    break
                
                self.analyzer.frame_count += 1
                
                processed_frame, score = self.analyzer.analyze_posture(frame)
                
                elapsed = time.time() - self.analyzer.start_time
                fps = self.analyzer.frame_count / elapsed if elapsed > 0 else 0
                
                cv2.putText(processed_frame, f"FPS: {fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if score is not None:
                    cv2.putText(processed_frame, f"Posture Score: {score:.0f}/100", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(processed_frame, f"Mode: {self.mode.value.upper()}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                cv2.putText(processed_frame, timestamp, 
                           (processed_frame.shape[1] - 120, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                cv2.imshow('Unified Posture Analysis System', processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("🚪 Quit requested")
                    break
                elif key == ord(' '):
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"posture_analysis_{timestamp_str}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    score_text = f" (Score: {score:.0f})" if score else ""
                    print(f"📸 Screenshot saved: {filename}{score_text}")
                elif key == ord('p') and self.mode == AnalysisMode.GPU_ACCELERATED:
                    self.analyzer.show_performance = not self.analyzer.show_performance
                    print(f"Performance overlay: {'ON' if self.analyzer.show_performance else 'OFF'}")
                
                if time.time() - last_status_time > 5:
                    score_text = f" | Score: {score:.0f}/100" if score else ""
                    print(f"📊 Running: {fps:.1f} FPS{score_text} | Frames: {self.analyzer.frame_count}")
                    last_status_time = time.time()
                    
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user (Ctrl+C)")
        except Exception as e:
            print(f"❌ Runtime error: {e}")
        finally:
            self._cleanup()
            return True
    
    def _cleanup(self):
        """Clean up resources"""
        print("\n🧹 Cleaning up...")
        
        self.analyzer.running = False
        self.camera_manager.release()
        cv2.destroyAllWindows()
        
        if self.analyzer.start_time:
            elapsed = time.time() - self.analyzer.start_time
            fps = self.analyzer.frame_count / elapsed if elapsed > 0 else 0
            
            print("📊 SESSION SUMMARY:")
            print(f"   Mode: {self.mode.value.upper()}")
            print(f"   Duration: {elapsed:.1f} seconds")
            print(f"   Frames processed: {self.analyzer.frame_count}")
            print(f"   Average FPS: {fps:.1f}")
            print(f"   Performance: {'Excellent' if fps > 25 else 'Good' if fps > 15 else 'Fair'}")
        
        print("✅ Unified posture analysis session completed")

def main():
    """Main entry point for unified posture analysis system"""
    parser = argparse.ArgumentParser(description='Unified Posture Analysis System')
    parser.add_argument('--mode', choices=['basic', 'advanced', 'gpu'], 
                       help='Force specific analysis mode')
    parser.add_argument('--config', help='Path to custom config file')
    
    args = parser.parse_args()
    
    mode = None
    if args.mode:
        mode = AnalysisMode(args.mode)
    
    try:
        system = PostureAnalysisSystem(mode=mode)
        success = system.run()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n🛑 Analysis stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
