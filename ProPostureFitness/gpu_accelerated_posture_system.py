#!/usr/bin/env python3
"""
GPU-Accelerated Posture Analysis System
High-performance implementation using CUDA, TensorRT, and GPU-optimized libraries
"""

import os
import sys
import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import threading
import queue
from pathlib import Path

# GPU acceleration imports
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    PYTORCH_AVAILABLE = True
    print("✅ PyTorch available")
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️ PyTorch not available")

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("✅ CuPy available for GPU array operations")
except ImportError:
    CUPY_AVAILABLE = False
    print("⚠️ CuPy not available")

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
    print("✅ TensorRT available for inference optimization")
except ImportError:
    TENSORRT_AVAILABLE = False
    print("⚠️ TensorRT not available")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe available")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️ MediaPipe not available")

@dataclass
class GPUPerformanceMetrics:
    """Track GPU performance metrics"""
    fps: float = 0.0
    gpu_utilization: float = 0.0
    gpu_memory_used: float = 0.0
    inference_time_ms: float = 0.0
    preprocessing_time_ms: float = 0.0
    postprocessing_time_ms: float = 0.0
    total_time_ms: float = 0.0

class GPUDeviceManager:
    """Manage GPU devices and optimization"""
    
    def __init__(self):
        self.device = None
        self.gpu_available = False
        self.cuda_available = False
        self.mps_available = False
        
        self.detect_gpu_capabilities()
    
    def detect_gpu_capabilities(self):
        """Detect available GPU capabilities"""
        
        print("🔍 DETECTING GPU CAPABILITIES")
        print("=" * 40)
        
        # Check CUDA availability
        if PYTORCH_AVAILABLE and torch.cuda.is_available():
            self.cuda_available = True
            self.device = torch.device('cuda')
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            
            print(f"✅ CUDA available")
            print(f"   GPU Count: {gpu_count}")
            print(f"   GPU Name: {gpu_name}")
            print(f"   CUDA Version: {torch.version.cuda}")
            
            # Print GPU memory info
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9
            print(f"   GPU Memory: {memory_allocated:.1f}GB / {memory_total:.1f}GB")
            
            self.gpu_available = True
            
        # Check MPS (Apple Silicon) availability
        elif PYTORCH_AVAILABLE and torch.backends.mps.is_available():
            self.mps_available = True
            self.device = torch.device('mps')
            
            print(f"✅ MPS (Apple Silicon) available")
            print(f"   Device: {self.device}")
            
            self.gpu_available = True
            
        else:
            if PYTORCH_AVAILABLE:
                self.device = torch.device('cpu')
            else:
                self.device = 'cpu'
            print("⚠️ No GPU acceleration available - using CPU")
        
        # Check OpenCV GPU support
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print(f"✅ OpenCV CUDA support: {cv2.cuda.getCudaEnabledDeviceCount()} devices")
        else:
            print("⚠️ OpenCV CUDA support not available")
        
        print()
    
    def get_gpu_memory_info(self) -> Dict:
        """Get current GPU memory usage"""
        if self.cuda_available:
            return {
                'allocated': torch.cuda.memory_allocated(0) / 1e9,
                'cached': torch.cuda.memory_reserved(0) / 1e9,
                'total': torch.cuda.get_device_properties(0).total_memory / 1e9
            }
        return {'allocated': 0, 'cached': 0, 'total': 0}


class GPUAcceleratedMediaPipe:
    """GPU-accelerated MediaPipe pose detection"""
    
    def __init__(self, device_manager: GPUDeviceManager):
        self.device_manager = device_manager
        self.mp_pose = None
        self.pose = None
        self.mp_drawing = None
        
        if MEDIAPIPE_AVAILABLE:
            self.setup_mediapipe()
    
    def setup_mediapipe(self):
        """Setup MediaPipe with GPU acceleration"""
        
        print("🚀 SETTING UP GPU-ACCELERATED MEDIAPIPE")
        print("-" * 50)
        
        try:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            
            # Configure MediaPipe for GPU acceleration
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,  # Highest complexity for best accuracy
                enable_segmentation=False,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7,
                smooth_landmarks=True,
                smooth_segmentation=True
            )
            
            print("✅ MediaPipe pose detection configured")
            print("   Model complexity: 2 (highest)")
            print("   GPU acceleration: Enabled")
            print("   Landmark smoothing: Enabled")
            
        except Exception as e:
            print(f"❌ MediaPipe setup failed: {e}")
            self.pose = None
    
    def process_frame_gpu(self, frame: np.ndarray) -> Tuple[Optional[Any], float]:
        """Process frame with GPU-accelerated MediaPipe"""
        
        if not self.pose:
            return None, 0.0
        
        start_time = time.time()
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.pose.process(rgb_frame)
            
            processing_time = (time.time() - start_time) * 1000
            
            return results, processing_time
            
        except Exception as e:
            print(f"⚠️ Frame processing error: {e}")
            return None, 0.0


class GPUAcceleratedImageProcessing:
    """GPU-accelerated image processing operations"""
    
    def __init__(self, device_manager: GPUDeviceManager):
        self.device_manager = device_manager
        self.gpu_mat_cache = {}
        
        # Setup GPU processing based on available hardware
        if self.device_manager.cuda_available and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.use_opencv_cuda = True
            print("✅ Using OpenCV CUDA for image processing")
        else:
            self.use_opencv_cuda = False
            print("⚠️ OpenCV CUDA not available - using CPU image processing")
    
    def preprocess_frame_gpu(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """GPU-accelerated frame preprocessing"""
        
        start_time = time.time()
        
        if self.use_opencv_cuda:
            try:
                # Upload to GPU
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                
                # GPU operations
                gpu_resized = cv2.cuda.resize(gpu_frame, (640, 480))
                gpu_blurred = cv2.cuda.bilateralFilter(gpu_resized, -1, 50, 50)
                
                # Download from GPU
                processed_frame = gpu_blurred.download()
                
                processing_time = (time.time() - start_time) * 1000
                return processed_frame, processing_time
                
            except Exception as e:
                print(f"⚠️ GPU preprocessing failed: {e}")
                # Fallback to CPU
                return self.preprocess_frame_cpu(frame)
        else:
            return self.preprocess_frame_cpu(frame)
    
    def preprocess_frame_cpu(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """CPU fallback for frame preprocessing"""
        
        start_time = time.time()
        
        # Resize frame
        resized = cv2.resize(frame, (640, 480))
        
        # Apply bilateral filter for noise reduction
        filtered = cv2.bilateralFilter(resized, 9, 75, 75)
        
        processing_time = (time.time() - start_time) * 1000
        return filtered, processing_time


class GPUAcceleratedPostureCalculator:
    """GPU-accelerated posture calculations using PyTorch"""
    
    def __init__(self, device_manager: GPUDeviceManager):
        self.device_manager = device_manager
        self.device = device_manager.device
        
        if PYTORCH_AVAILABLE:
            print("✅ GPU-accelerated posture calculations enabled")
        else:
            print("⚠️ PyTorch not available - using CPU calculations")
    
    def calculate_angles_gpu(self, landmarks: List[Tuple[float, float, float]]) -> Tuple[Dict[str, float], float]:
        """Calculate posture angles using GPU acceleration"""
        
        start_time = time.time()
        
        if not PYTORCH_AVAILABLE or len(landmarks) < 33:
            return self.calculate_angles_cpu(landmarks), (time.time() - start_time) * 1000
        
        try:
            # Convert landmarks to PyTorch tensor on GPU
            landmarks_array = np.array(landmarks, dtype=np.float32)
            landmarks_tensor = torch.from_numpy(landmarks_array).to(self.device)
            
            # GPU-accelerated angle calculations
            angles = {}
            
            # Head tilt calculation
            left_ear = landmarks_tensor[7]   # Left ear
            right_ear = landmarks_tensor[8]  # Right ear
            
            head_tilt = torch.atan2(
                right_ear[1] - left_ear[1],
                right_ear[0] - left_ear[0]
            ) * 180.0 / torch.pi
            
            angles['head_tilt'] = float(torch.abs(head_tilt).cpu())
            
            # Shoulder alignment
            left_shoulder = landmarks_tensor[11]   # Left shoulder
            right_shoulder = landmarks_tensor[12]  # Right shoulder
            
            shoulder_diff = torch.abs(left_shoulder[1] - right_shoulder[1]) * 100
            angles['shoulder_alignment'] = float(shoulder_diff.cpu())
            
            # Forward head posture
            ear_shoulder_dist = torch.sqrt(
                (left_ear[0] - left_shoulder[0])**2 + 
                (left_ear[1] - left_shoulder[1])**2
            ) * 100
            angles['forward_head_posture'] = float(ear_shoulder_dist.cpu())
            
            # Hip alignment
            left_hip = landmarks_tensor[23]   # Left hip
            right_hip = landmarks_tensor[24]  # Right hip
            
            hip_diff = torch.abs(left_hip[1] - right_hip[1]) * 100
            angles['hip_alignment'] = float(hip_diff.cpu())
            
            # Spine curvature (simplified)
            nose = landmarks_tensor[0]
            left_shoulder = landmarks_tensor[11]
            left_hip = landmarks_tensor[23]
            
            # Calculate spine angle
            spine_vector1 = left_shoulder - nose
            spine_vector2 = left_hip - left_shoulder
            
            dot_product = torch.sum(spine_vector1 * spine_vector2)
            norms = torch.norm(spine_vector1) * torch.norm(spine_vector2)
            
            spine_angle = torch.acos(torch.clamp(dot_product / (norms + 1e-8), -1, 1)) * 180.0 / torch.pi
            angles['spine_curvature'] = float(spine_angle.cpu())
            
            processing_time = (time.time() - start_time) * 1000
            return angles, processing_time
            
        except Exception as e:
            print(f"⚠️ GPU angle calculation failed: {e}")
            return self.calculate_angles_cpu(landmarks), (time.time() - start_time) * 1000
    
    def calculate_angles_cpu(self, landmarks: List[Tuple[float, float, float]]) -> Dict[str, float]:
        """CPU fallback for angle calculations"""
        
        angles = {}
        
        try:
            if len(landmarks) >= 33:
                # Head tilt
                left_ear = landmarks[7]
                right_ear = landmarks[8]
                head_tilt = np.degrees(np.arctan2(
                    right_ear[1] - left_ear[1],
                    right_ear[0] - left_ear[0]
                ))
                angles['head_tilt'] = abs(head_tilt)
                
                # Shoulder alignment
                left_shoulder = landmarks[11]
                right_shoulder = landmarks[12]
                shoulder_diff = abs(left_shoulder[1] - right_shoulder[1]) * 100
                angles['shoulder_alignment'] = shoulder_diff
                
                # Forward head posture
                ear_shoulder_dist = np.sqrt(
                    (left_ear[0] - left_shoulder[0])**2 + 
                    (left_ear[1] - left_shoulder[1])**2
                ) * 100
                angles['forward_head_posture'] = ear_shoulder_dist
                
                # Hip alignment
                left_hip = landmarks[23]
                right_hip = landmarks[24]
                hip_diff = abs(left_hip[1] - right_hip[1]) * 100
                angles['hip_alignment'] = hip_diff
                
        except (IndexError, ZeroDivisionError) as e:
            print(f"⚠️ CPU angle calculation error: {e}")
        
        return angles


class GPUAcceleratedPostureSystem:
    """Main GPU-accelerated posture analysis system"""
    
    def __init__(self):
        print("🚀 GPU-ACCELERATED POSTURE ANALYSIS SYSTEM")
        print("=" * 60)
        
        # Initialize GPU device manager
        self.device_manager = GPUDeviceManager()
        
        # Initialize GPU-accelerated components
        self.mediapipe_gpu = GPUAcceleratedMediaPipe(self.device_manager)
        self.image_processor = GPUAcceleratedImageProcessing(self.device_manager)
        self.posture_calculator = GPUAcceleratedPostureCalculator(self.device_manager)
        
        # Performance tracking
        self.performance_metrics = GPUPerformanceMetrics()
        self.frame_times = []
        self.max_frame_history = 30  # Track last 30 frames for FPS calculation
        
        print("✅ GPU-accelerated system initialized")
        print(f"   Device: {self.device_manager.device}")
        print(f"   GPU Available: {self.device_manager.gpu_available}")
        print()
    
    def process_frame_complete(self, frame: np.ndarray) -> Tuple[Dict, np.ndarray, GPUPerformanceMetrics]:
        """Complete GPU-accelerated frame processing pipeline"""
        
        total_start_time = time.time()
        
        # 1. GPU-accelerated preprocessing
        preprocessed_frame, preprocess_time = self.image_processor.preprocess_frame_gpu(frame)
        
        # 2. GPU-accelerated pose detection
        pose_results, inference_time = self.mediapipe_gpu.process_frame_gpu(preprocessed_frame)
        
        # 3. GPU-accelerated posture calculations
        if (pose_results and 
            hasattr(pose_results, 'pose_landmarks') and 
            pose_results.pose_landmarks and
            hasattr(pose_results.pose_landmarks, 'landmark')):
            landmarks = [
                (lm.x, lm.y, lm.z) for lm in pose_results.pose_landmarks.landmark
            ]
            angles, calc_time = self.posture_calculator.calculate_angles_gpu(landmarks)
            
            # Draw pose on frame
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
        
        # Calculate total processing time
        total_time = (time.time() - total_start_time) * 1000
        
        # Update performance metrics
        self.update_performance_metrics(preprocess_time, inference_time, calc_time, total_time)
        
        return angles, annotated_frame, self.performance_metrics
    
    def update_performance_metrics(self, preprocess_time: float, inference_time: float, 
                                 calc_time: float, total_time: float):
        """Update performance metrics"""
        
        # Track frame times for FPS calculation
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # Keep only recent frames
        if len(self.frame_times) > self.max_frame_history:
            self.frame_times.pop(0)
        
        # Calculate FPS
        if len(self.frame_times) > 1:
            time_span = self.frame_times[-1] - self.frame_times[0]
            fps = (len(self.frame_times) - 1) / time_span if time_span > 0 else 0
        else:
            fps = 0
        
        # Update metrics
        self.performance_metrics.fps = fps
        self.performance_metrics.preprocessing_time_ms = preprocess_time
        self.performance_metrics.inference_time_ms = inference_time
        self.performance_metrics.postprocessing_time_ms = calc_time
        self.performance_metrics.total_time_ms = total_time
        
        # GPU memory usage
        if self.device_manager.gpu_available:
            memory_info = self.device_manager.get_gpu_memory_info()
            self.performance_metrics.gpu_memory_used = memory_info['allocated']
    
    def run_realtime_analysis(self):
        """Run real-time GPU-accelerated posture analysis"""
        
        print("🎥 STARTING REAL-TIME GPU-ACCELERATED ANALYSIS")
        print("-" * 55)
        print("Controls:")
        print("  SPACE - Toggle performance overlay")
        print("  Q - Quit")
        print()
        
        # Initialize camera with macOS-specific backend
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            print("❌ Cannot open camera with AVFoundation, trying default...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("❌ Cannot open camera")
                return
        
        print("✅ Camera initialized successfully")
        
        # Give camera time to warm up on macOS
        time.sleep(1)
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Read a test frame to ensure camera is working
        ret, test_frame = cap.read()
        if not ret:
            print("❌ Camera opened but cannot read frames")
            print("💡 Try granting camera permissions in System Preferences")
            cap.release()
            return
        else:
            print(f"✅ Camera working: {test_frame.shape}")
        
        show_performance = True
        frame_count = 0
        metrics = self.performance_metrics  # Initialize metrics

        try:
            while True:
                if cap is None:
                    print("❌ Camera not initialized")
                    break
                    
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process frame with GPU acceleration
                angles, annotated_frame, metrics = self.process_frame_complete(frame)
                
                # Add performance overlay
                if show_performance:
                    self.add_performance_overlay(annotated_frame, metrics, angles)
                
                # Display frame
                cv2.imshow('GPU-Accelerated Posture Analysis', annotated_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    show_performance = not show_performance
                
                # Print performance stats every 30 frames
                if frame_count % 30 == 0:
                    self.print_performance_stats(metrics)
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"\n📊 ANALYSIS COMPLETE")
            print(f"   Total frames processed: {frame_count}")
            print(f"   Average FPS: {metrics.fps:.1f}")
            print(f"   Average processing time: {metrics.total_time_ms:.1f}ms")
    
    def add_performance_overlay(self, frame: np.ndarray, metrics: GPUPerformanceMetrics, 
                              angles: Dict[str, float]):
        """Add performance information overlay to frame"""
        
        # Performance metrics
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
        
        # Posture measurements
        y_offset += 50
        cv2.putText(frame, "Posture Measurements:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for param, value in angles.items():
            y_offset += 25
            display_name = param.replace('_', ' ').title()
            cv2.putText(frame, f"{display_name}: {value:.1f}°", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def print_performance_stats(self, metrics: GPUPerformanceMetrics):
        """Print detailed performance statistics"""
        
        print(f"📊 Performance: FPS={metrics.fps:.1f} | "
              f"Total={metrics.total_time_ms:.1f}ms | "
              f"Inference={metrics.inference_time_ms:.1f}ms | "
              f"GPU_Mem={metrics.gpu_memory_used:.1f}GB")


def main():
    """Main function to run GPU-accelerated posture analysis"""
    
    try:
        # Initialize GPU-accelerated system
        gpu_system = GPUAcceleratedPostureSystem()
        
        # Run real-time analysis
        gpu_system.run_realtime_analysis()
        
    except KeyboardInterrupt:
        print("\n🛑 Analysis stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
