#!/bin/bash


echo "🏃‍♂️ FitlifePostureProApp VSCode Setup"
echo "======================================"

if [ ! -f "FitlifePostureProApp.py" ]; then
    echo "❌ Error: Please run this script from the ProPostureFitness directory"
    echo "   cd path/to/posture/ProPostureFitness"
    exit 1
fi

echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "   Virtual environment already exists, removing old one..."
    rm -rf venv
fi

python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment"
    exit 1
fi

echo "🔄 Activating virtual environment..."
source venv/bin/activate

echo "⬆️ Upgrading pip..."
python -m pip install --upgrade pip

echo "📚 Installing core dependencies..."
pip install numpy>=1.21.0
pip install opencv-python>=4.8.0
pip install Pillow>=10.0.0

echo "🎯 Installing MediaPipe (with fallback handling)..."
pip install mediapipe>=0.10.0
if [ $? -ne 0 ]; then
    echo "⚠️ MediaPipe installation failed, trying alternative version..."
    pip install mediapipe==0.9.3.0
    if [ $? -ne 0 ]; then
        echo "⚠️ MediaPipe not available for your system"
        echo "   The app will run in Basic mode only"
    fi
fi

echo "🚀 Installing PyTorch (optional, for GPU acceleration)..."
pip install torch torchvision
if [ $? -ne 0 ]; then
    echo "⚠️ PyTorch installation failed"
    echo "   GPU acceleration will not be available"
fi

echo "📋 Installing remaining dependencies..."
pip install -r requirements.txt --ignore-installed
if [ $? -ne 0 ]; then
    echo "⚠️ Some dependencies from requirements.txt failed to install"
    echo "   Core functionality should still work"
fi

echo ""
echo "✅ Setup completed!"
echo ""
echo "🎯 Next Steps:"
echo "1. Open VSCode in the project root:"
echo "   code ."
echo ""
echo "2. In VSCode terminal, activate the virtual environment:"
echo "   source ProPostureFitness/venv/bin/activate"
echo ""
echo "3. Run the application:"
echo "   python FitlifePostureProApp.py"
echo ""
echo "📊 Available modes:"
echo "   python FitlifePostureProApp.py --mode basic     # Face detection only"
echo "   python FitlifePostureProApp.py --mode advanced  # Full pose analysis"
echo "   python FitlifePostureProApp.py --mode gpu       # GPU accelerated"
echo ""
echo "🔧 VSCode Extensions (recommended):"
echo "   - Python (by Microsoft)"
echo "   - Python Debugger"
echo "   - Pylance"
echo ""
echo "📷 Camera Access:"
echo "   Make sure your camera is not being used by other applications"
echo "   Grant camera permissions when prompted"
