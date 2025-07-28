#!/usr/bin/env python3
"""
FitlifePostureProApp VSCode Setup Script (Python version)
Cross-platform setup with dependency fallbacks
"""

import os
import sys
import subprocess
import platform

def run_command(command, description=""):
    """Run a command and return success status"""
    if description:
        print(f"   {description}...")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            return True
        else:
            print(f"   ⚠️ Warning: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("   FitlifePostureProApp requires Python 3.8 or higher")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def setup_virtual_environment():
    """Create and activate virtual environment"""
    print("📦 Setting up virtual environment...")
    
    if os.path.exists("venv"):
        print("   Removing existing virtual environment...")
        if platform.system() == "Windows":
            run_command("rmdir /s /q venv")
        else:
            run_command("rm -rf venv")
    
    if not run_command(f"{sys.executable} -m venv venv", "Creating virtual environment"):
        return False
    
    print("✅ Virtual environment created successfully")
    return True

def get_pip_command():
    """Get the correct pip command for the virtual environment"""
    if platform.system() == "Windows":
        return "venv\\Scripts\\python -m pip"
    else:
        return "venv/bin/python -m pip"

def install_dependencies():
    """Install dependencies with fallbacks"""
    pip_cmd = get_pip_command()
    
    print("📚 Installing dependencies...")
    
    run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip")
    
    core_deps = [
        "numpy>=1.21.0",
        "opencv-python>=4.8.0", 
        "Pillow>=10.0.0"
    ]
    
    for dep in core_deps:
        if not run_command(f"{pip_cmd} install {dep}", f"Installing {dep}"):
            print(f"   ❌ Failed to install {dep}")
            return False
    
    print("🎯 Installing MediaPipe...")
    if not run_command(f"{pip_cmd} install mediapipe>=0.10.0", "Installing MediaPipe latest"):
        print("   Trying fallback MediaPipe version...")
        if not run_command(f"{pip_cmd} install mediapipe==0.9.3.0", "Installing MediaPipe 0.9.3.0"):
            print("   ⚠️ MediaPipe not available - Basic mode only")
    
    print("🚀 Installing PyTorch (optional)...")
    if not run_command(f"{pip_cmd} install torch torchvision", "Installing PyTorch"):
        print("   ⚠️ PyTorch not available - No GPU acceleration")
    
    if os.path.exists("requirements.txt"):
        print("📋 Installing from requirements.txt...")
        run_command(f"{pip_cmd} install -r requirements.txt", "Installing requirements.txt")
    
    return True

def test_installation():
    """Test that the installation works"""
    print("🧪 Testing installation...")
    
    if platform.system() == "Windows":
        python_cmd = "venv\\Scripts\\python"
    else:
        python_cmd = "venv/bin/python"
    
    test_code = '''
import sys
try:
    import cv2
    print("✅ OpenCV available")
except ImportError:
    print("❌ OpenCV not available")
    sys.exit(1)

try:
    import mediapipe
    print("✅ MediaPipe available")
except ImportError:
    print("⚠️ MediaPipe not available (Basic mode only)")

try:
    import torch
    print("✅ PyTorch available")
except ImportError:
    print("⚠️ PyTorch not available (No GPU acceleration)")

print("✅ Installation test completed")
'''
    
    return run_command(f'{python_cmd} -c "{test_code}"', "Testing imports")

def print_instructions():
    """Print final setup instructions"""
    print("\n" + "="*50)
    print("✅ FitlifePostureProApp Setup Complete!")
    print("="*50)
    
    print("\n🎯 VSCode Setup Instructions:")
    print("1. Open VSCode in the project root:")
    print("   code .")
    
    print("\n2. Install recommended VSCode extensions:")
    print("   - Python (by Microsoft)")
    print("   - Python Debugger") 
    print("   - Pylance")
    
    print("\n3. Select Python interpreter:")
    print("   - Press Ctrl+Shift+P (Cmd+Shift+P on Mac)")
    print("   - Type 'Python: Select Interpreter'")
    if platform.system() == "Windows":
        print("   - Choose: ./ProPostureFitness/venv/Scripts/python.exe")
    else:
        print("   - Choose: ./ProPostureFitness/venv/bin/python")
    
    print("\n4. Run the application:")
    print("   python FitlifePostureProApp.py")
    
    print("\n📊 Available modes:")
    print("   python FitlifePostureProApp.py --mode basic     # Face detection")
    print("   python FitlifePostureProApp.py --mode advanced  # Full pose analysis")
    print("   python FitlifePostureProApp.py --mode gpu       # GPU accelerated")
    
    print("\n📷 Camera Setup:")
    print("   - Ensure camera is not used by other apps")
    print("   - Grant camera permissions when prompted")
    print("   - Test with: python -c 'import cv2; print(cv2.VideoCapture(0).read())'")

def main():
    """Main setup function"""
    print("🏃‍♂️ FitlifePostureProApp VSCode Setup")
    print("="*40)
    
    if not os.path.exists("FitlifePostureProApp.py"):
        print("❌ Error: FitlifePostureProApp.py not found")
        print("   Please run this script from the ProPostureFitness directory")
        print("   cd path/to/posture/ProPostureFitness")
        return False
    
    if not check_python_version():
        return False
    
    if not setup_virtual_environment():
        return False
    
    if not install_dependencies():
        return False
    
    test_installation()
    
    print_instructions()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
