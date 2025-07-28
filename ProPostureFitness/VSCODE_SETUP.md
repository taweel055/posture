# VSCode Setup Guide for FitlifePostureProApp

## Quick Setup (Automated)

### Option 1: Bash Script (macOS/Linux)
```bash
chmod +x setup_vscode.sh
./setup_vscode.sh
```

### Option 2: Python Script (Cross-platform)
```bash
python3 setup_vscode.py
```

## Manual Setup

### 1. Clone Repository
```bash
git clone https://github.com/taweel055/posture.git
cd posture/ProPostureFitness
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install core dependencies
pip install numpy>=1.21.0
pip install opencv-python>=4.8.0
pip install Pillow>=10.0.0

# Install MediaPipe (with fallback)
pip install mediapipe>=0.10.0
# If that fails:
pip install mediapipe==0.9.3.0

# Install PyTorch (optional, for GPU)
pip install torch torchvision

# Install remaining dependencies
pip install -r requirements.txt
```

### 4. Open in VSCode
```bash
code .
```

### 5. Configure VSCode

#### Install Extensions:
- Python (by Microsoft)
- Python Debugger
- Pylance

#### Select Python Interpreter:
1. Press `Ctrl+Shift+P` (`Cmd+Shift+P` on Mac)
2. Type "Python: Select Interpreter"
3. Choose: `./ProPostureFitness/venv/bin/python` (or `venv\Scripts\python.exe` on Windows)

### 6. Run Application
```bash
# Auto-detect best mode
python FitlifePostureProApp.py

# Specific modes
python FitlifePostureProApp.py --mode basic
python FitlifePostureProApp.py --mode advanced
python FitlifePostureProApp.py --mode gpu
```

## Troubleshooting

### MediaPipe Issues on macOS ARM64
If MediaPipe fails to install:
```bash
# Try specific version
pip install mediapipe==0.9.3.0

# Or use conda
conda install -c conda-forge mediapipe
```

### Camera Access Issues
- Ensure camera isn't used by other applications
- Grant camera permissions when prompted
- Test camera: `python -c "import cv2; print(cv2.VideoCapture(0).read())"`

### Virtual Environment Issues
If you get "externally-managed-environment" error:
```bash
# Use virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Or use --break-system-packages (not recommended)
pip install --break-system-packages -r requirements.txt
```

## Application Controls

### FitlifePostureProApp Controls:
- **`SPACE`** - Take screenshot
- **`s`** - Save current analysis
- **`r`** - Reset session
- **`h`** - Show help
- **`q`** - Quit application

### Analysis Modes:
- **Basic**: Face detection only (minimal dependencies)
- **Advanced**: Full pose analysis with MediaPipe
- **GPU**: GPU-accelerated analysis with PyTorch

## System Requirements

- Python 3.8 or higher
- Webcam or camera device
- 4GB RAM minimum (8GB recommended for GPU mode)
- macOS 10.14+, Windows 10+, or Linux

## Dependencies

### Core (Required):
- opencv-python>=4.8.0
- numpy>=1.21.0
- Pillow>=10.0.0

### Optional:
- mediapipe>=0.10.0 (for Advanced mode)
- torch, torchvision (for GPU mode)

## Support

If you encounter issues:
1. Check camera permissions
2. Verify Python version (3.8+)
3. Ensure virtual environment is activated
4. Try Basic mode first: `python FitlifePostureProApp.py --mode basic`
