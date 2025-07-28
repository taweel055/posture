#!/usr/bin/env python3
"""
Integrated Detailed Assessment System
Combines automated posture assessment with detailed report generation
"""

import cv2
import time
from datetime import datetime
from pathlib import Path

# Import existing systems
from automated_enhanced_camera_app import AutomatedEnhancedCameraApp
from detailed_assessment_report_generator import DetailedReportGenerator, export_html_report

class IntegratedDetailedAssessmentApp(AutomatedEnhancedCameraApp):
    """Enhanced camera app with integrated detailed reporting"""
    
    def __init__(self):
        super().__init__()
        self.detailed_report_generator = DetailedReportGenerator()
        self.detailed_reports_dir = Path("~/Documents/Posture check test/detailed_reports").expanduser()
        self.detailed_reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhanced controls
        self.show_detailed_report_preview = False
        self.last_detailed_report = None
        
        print("📊 Integrated Detailed Assessment System Ready")
        print("🎯 Press 'd' for detailed assessment report")
        print("📋 Press 'v' to view last detailed report")
    
    def run_camera_mode(self):
        """Run camera mode with detailed assessment integration"""
        print("\n🎯 INTEGRATED DETAILED POSTURE ASSESSMENT CAMERA")
        print("=" * 70)
        print("⏱️ 10-second countdown before assessment becomes available")
        print("🤖 Press 'a' to start/stop automated assessment")
        print("📊 Press 'd' for detailed assessment report")
        print("📋 Press 'v' to view last detailed report")
        print("📐 Press 'g' to toggle grid overlay")
        print("🎯 Press 'f' to set per-frame assessment (30 FPS)")
        print("📱 Press 'o' to toggle orientation (Portrait/Landscape)")
        print("🔧 Press '+/-' to adjust automation frequency (0.1-30 FPS)")
        print("❓ Press 'h' for help, 'q' to quit")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot access camera")
            return self.run_demo_mode()
        
        # Start countdown
        self.countdown_start_time = time.time()
        
        try:
            while self.running:
                ret, frame = cap.read()
                
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                # Store original frame for assessment
                original_frame = frame.copy()
                
                # Process frame for pose detection and automated assessment
                processed_frame, pose_landmarks = self.process_frame_for_display(frame)
                self.current_frame = processed_frame
                self.pose_landmarks = pose_landmarks
                
                # Create display frame with overlays
                display_frame = self.create_display_frame(processed_frame)
                
                # Display frame
                if display_frame is not None:
                    window_title = f'Integrated Detailed Assessment - {self.orientation_mode.title()} Mode'
                    cv2.imshow(window_title, display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("🚪 Quit requested")
                    break
                elif key == ord(' '):  # Spacebar for manual assessment
                    if self.assessment_enabled and not self.assessment_in_progress:
                        score = self.take_manual_posture_reading(original_frame)
                        if score is not None:
                            print(f"📊 Manual Assessment Result: {score:.1f} ({self.orientation_mode})")
                    elif not self.assessment_enabled:
                        remaining = self.update_countdown()
                        if remaining and remaining > 0:
                            print(f"⏱️ Please wait {int(remaining)} more seconds")
                elif key == ord('d'):  # Detailed assessment report
                    if self.assessment_enabled and pose_landmarks:
                        self.generate_detailed_assessment_report(pose_landmarks, original_frame)
                    else:
                        print("📊 Take a posture reading first or wait for countdown")
                elif key == ord('v'):  # View last detailed report
                    if self.last_detailed_report:
                        self.open_detailed_report(self.last_detailed_report)
                    else:
                        print("📊 No detailed report available - press 'd' to generate one")
                elif key == ord('a'):  # Toggle automation
                    if self.assessment_enabled:
                        self.automation_engine.toggle_automation()
                    else:
                        print("⏱️ Wait for countdown to complete before starting automation")
                elif key == ord('g'):  # Toggle grid overlay
                    self.grid_overlay.toggle_grid()
                elif key == ord('h'):
                    self.show_help = not self.show_help
                    print(f"❓ Help overlay: {'ON' if self.show_help else 'OFF'}")
                elif key == ord('o'):  # Orientation toggle
                    self.toggle_orientation()
                elif key == ord('p'):  # Toggle pose landmarks
                    self.show_pose = not self.show_pose
                    print(f"🤖 Pose landmarks: {'ON' if self.show_pose else 'OFF'}")
                elif key == ord('s'):  # Toggle automation stats
                    self.show_automation_stats = not self.show_automation_stats
                    print(f"📊 Automation stats: {'ON' if self.show_automation_stats else 'OFF'}")
                elif key == ord('=') or key == ord('+'):  # Increase frequency
                    current_freq = self.automation_engine.assessment_frequency
                    if current_freq < 1.0:
                        new_freq = current_freq + 0.1
                    elif current_freq < 5.0:
                        new_freq = current_freq + 0.5
                    elif current_freq < 30.0:
                        new_freq = min(30.0, current_freq + 5.0)
                    else:
                        new_freq = 30.0
                    self.automation_engine.update_assessment_frequency(new_freq)
                elif key == ord('-'):  # Decrease frequency
                    current_freq = self.automation_engine.assessment_frequency
                    if current_freq > 5.0:
                        new_freq = max(5.0, current_freq - 5.0)
                    elif current_freq > 1.0:
                        new_freq = max(1.0, current_freq - 0.5)
                    else:
                        new_freq = max(0.1, current_freq - 0.1)
                    self.automation_engine.update_assessment_frequency(new_freq)
                elif key == ord('f'):  # Set per-frame assessment (30 FPS)
                    self.automation_engine.update_assessment_frequency(30.0)
                    print("🎯 Per-frame assessment activated (30 FPS)")
                elif key == ord('1'):  # Set 1 FPS
                    self.automation_engine.update_assessment_frequency(1.0)
                elif key == ord('5'):  # Set 5 FPS
                    self.automation_engine.update_assessment_frequency(5.0)
                elif key == ord('0'):  # Set 10 FPS
                    self.automation_engine.update_assessment_frequency(10.0)
                elif key == ord('c'):  # Clear alerts
                    self.automation_alerts.clear()
                    print("🔄 Alerts cleared")
                elif key == ord('r'):  # Toggle detailed report display
                    if self.current_report is not None:
                        self.show_detailed_report = not self.show_detailed_report
                        print(f"📊 Detailed report: {'ON' if self.show_detailed_report else 'OFF'}")
                    else:
                        print("📊 No report available - take an assessment first")
                
        except KeyboardInterrupt:
            print("\n⚠️ Application interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("🧹 Cleanup completed")
    
    def generate_detailed_assessment_report(self, pose_landmarks, original_frame):
        """Generate comprehensive detailed assessment report"""
        print("📊 Generating detailed assessment report...")
        
        try:
            # Generate detailed report
            detailed_report = self.detailed_report_generator.generate_detailed_report(
                pose_landmarks=pose_landmarks,
                orientation_mode=self.orientation_mode,
                session_data=self.automation_engine.get_session_stats(),
                pytorch_enhanced=self.posture_analyzer.use_pytorch
            )
            
            # Save JSON report
            json_file = self.detailed_reports_dir / f"detailed_report_{detailed_report.assessment_id}.json"
            with open(json_file, 'w') as f:
                import json
                from dataclasses import asdict
                json.dump(asdict(detailed_report), f, indent=2)
            
            # Save HTML report
            html_file = self.detailed_reports_dir / f"detailed_report_{detailed_report.assessment_id}.html"
            export_html_report(detailed_report, str(html_file))
            
            # Store reference
            self.last_detailed_report = str(html_file)
            
            # Display summary
            print(f"✅ Detailed assessment report generated:")
            print(f"   📄 JSON: {json_file}")
            print(f"   🌐 HTML: {html_file}")
            print(f"🎯 Overall Score: {detailed_report.overall_score:.1f}/100 ({detailed_report.posture_grade})")
            print(f"📋 Issues Identified: {len(detailed_report.identified_issues)}")
            print(f"💡 Priority Recommendations: {len(detailed_report.priority_recommendations)}")
            print(f"🏃‍♂️ Exercise Program: {len(detailed_report.exercise_program)} items")
            print(f"📅 Next Assessment: {detailed_report.next_assessment_recommendation}")
            print(f"🔄 Monitoring Frequency: {detailed_report.monitoring_frequency}")
            print(f"📊 Press 'v' to open detailed report in browser")
            
        except Exception as e:
            print(f"❌ Failed to generate detailed report: {e}")
    
    def open_detailed_report(self, report_path):
        """Open detailed report in browser"""
        try:
            import webbrowser
            webbrowser.open(f"file://{report_path}")
            print(f"🌐 Opened detailed report in browser: {Path(report_path).name}")
        except Exception as e:
            print(f"❌ Failed to open report: {e}")
    
    def add_detailed_controls_overlay(self, frame):
        """Add detailed assessment controls to overlay"""
        if frame is None:
            return frame
        
        # Enhanced controls panel
        panel_width, panel_height = 500, 200
        x = 20
        y = frame.shape[0] - panel_height - 20
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Border color based on assessment availability
        border_color = (0, 255, 0) if self.assessment_enabled else (128, 128, 128)
        cv2.rectangle(frame, (x, y), (x + panel_width, y + panel_height), border_color, 2)
        
        # Title
        cv2.putText(frame, "🎯 INTEGRATED DETAILED ASSESSMENT", (x + 10, y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Automation status
        auto_status = "ACTIVE" if self.automation_engine.enabled else "STOPPED"
        auto_color = (0, 255, 0) if self.automation_engine.enabled else (0, 0, 255)
        cv2.putText(frame, f"Automation: {auto_status}", (x + 10, y + 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, auto_color, 1)
        
        # Frequency display
        if self.automation_engine.assessment_frequency >= 30.0:
            freq_text = "Frequency: EVERY FRAME (30 FPS)"
            freq_color = (0, 255, 0)
        else:
            freq_text = f"Frequency: {self.automation_engine.assessment_frequency:.1f}/sec"
            freq_color = (255, 255, 0)
        cv2.putText(frame, freq_text, (x + 10, y + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, freq_color, 1)
        
        # Last report status
        if self.last_detailed_report:
            cv2.putText(frame, "Last Report: Available (press 'v' to view)", (x + 10, y + 105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            cv2.putText(frame, "Last Report: None (press 'd' to generate)", (x + 10, y + 105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Controls
        cv2.putText(frame, "d - Detailed Report  |  v - View Report  |  a - Auto  |  f - 30FPS", (x + 10, y + 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, "SPACE - Manual  |  g - Grid  |  o - Orient  |  h - Help  |  q - Quit", (x + 10, y + 155), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, "+/- Freq  |  1,5,0 Presets  |  s - Stats  |  c - Clear Alerts", (x + 10, y + 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def create_display_frame(self, frame):
        """Create display frame with all overlays including detailed controls"""
        if frame is None:
            return None
        
        display_frame = frame.copy()
        
        # Add grid overlay first (behind everything)
        display_frame = self.grid_overlay.draw_grid(display_frame)
        
        # Add countdown overlay
        remaining_time = self.update_countdown()
        if remaining_time is not None and remaining_time > 0:
            display_frame = self.add_countdown_overlay(display_frame, remaining_time)
        
        # Add pose landmarks
        if self.show_pose and self.pose_landmarks:
            display_frame = self.add_pose_landmarks(display_frame)
        
        # Add automation overlay
        display_frame = self.add_automation_overlay(display_frame)
        
        # Add real-time alerts
        display_frame = self.add_real_time_alerts_overlay(display_frame)
        
        # Add detailed controls overlay (replaces standard controls)
        display_frame = self.add_detailed_controls_overlay(display_frame)
        
        # Add assessment result overlay
        if self.show_assessment_result and hasattr(self, 'last_assessment_score') and self.last_assessment_score is not None:
            display_frame = self.add_assessment_result_overlay(display_frame)
        
        # Add detailed report overlay
        if self.show_detailed_report and self.current_report is not None:
            display_frame = self.add_detailed_report_overlay(display_frame)
        
        # Add help overlay
        if self.show_help:
            display_frame = self.add_help_overlay(display_frame)
        
        return display_frame

def main():
    """Main function for integrated detailed assessment"""
    app = IntegratedDetailedAssessmentApp()
    
    print("🎯 INTEGRATED DETAILED POSTURE ASSESSMENT APPLICATION")
    print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    try:
        app.run_camera_mode()
    except Exception as e:
        print(f"❌ Application error: {e}")
    finally:
        # Finalize automated logging session
        app.automated_logger.finalize_session()
        print("👋 Integrated Detailed Assessment Application closed")

if __name__ == "__main__":
    main()
