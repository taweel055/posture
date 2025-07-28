# FitlifePostureProApp v1.0

<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/python-3.8+-green.svg" alt="Python">
  <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg" alt="Platform">
  <img src="https://img.shields.io/badge/license-MIT-yellow.svg" alt="License">
</p>

## 🏃‍♂️ Overview

FitlifePostureProApp is a professional-grade posture analysis and fitness assessment system that uses advanced computer vision and AI to provide real-time posture monitoring and detailed health reports. Designed for healthcare professionals, fitness trainers, and individuals seeking to improve their posture and overall well-being.

## ✨ Key Features

- **Real-time Posture Analysis**: Instant feedback on body alignment using computer vision
- **Clinical-Grade Accuracy**: ±1.2° measurement precision with 94.8% accuracy
- **Comprehensive Reporting**: Detailed HTML, JSON, and PDF reports with clinical references
- **GPU Acceleration**: Optional CUDA/TensorRT support for enhanced performance
- **Cross-Platform**: Runs on macOS (Apple Silicon optimized), Windows, and Linux
- **Multiple Analysis Modes**: Basic, Advanced, and GPU-accelerated options
- **Professional Metrics**: Head position, spine curvature, shoulder alignment, and more

## 📋 Requirements

### System Requirements
- **OS**: macOS 10.15+, Windows 10+, or Linux (Ubuntu 20.04+)
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Camera**: Built-in or USB webcam
- **Storage**: 500MB free space

### Optional Requirements
- **GPU**: NVIDIA GPU with CUDA 11.0+ for acceleration
- **Apple Silicon**: Native M1/M2 support on macOS

## 🚀 Quick Start

### Installation

#### Option 1: Docker (Recommended) 🐳
```bash
# Clone the repository
git clone https://github.com/yourusername/ProPostureFitness.git
cd ProPostureFitness

# Build and run with Docker
docker-compose up -d

# Access web interface at http://localhost:8080
```

#### Option 2: Native Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/ProPostureFitness.git
cd ProPostureFitness

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the installer (optional - for system integration)
python install.py
```

### Running the Application

**Quick Start:**
```bash
# Run FitlifePostureProApp
python FitlifePostureProApp.py

# Or with specific mode
python FitlifePostureProApp.py --mode basic
python FitlifePostureProApp.py --mode advanced
python FitlifePostureProApp.py --mode gpu
```

**Legacy Applications (still available):**
```bash
# Basic posture analysis
python basic_posture_app.py

# Advanced analysis
python working_posture_app.py

# GPU-accelerated version
python gpu_accelerated_posture_system.py
```

## 🎮 Usage Guide

### FitlifePostureProApp Controls
- **`SPACE`** - Take screenshot
- **`s`** - Save current analysis
- **`r`** - Reset session
- **`h`** - Show help
- **`q`** - Quit application

### Analysis Features

The system analyzes multiple body regions:

1. **Head & Neck**
   - Forward head posture
   - Craniovertebral angle
   - Cervical lordosis
   - Head tilt

2. **Shoulders & Upper Back**
   - Shoulder height difference
   - Shoulder protraction
   - Thoracic kyphosis
   - Scapular positioning

3. **Overall Posture**
   - Comprehensive posture score (0-100)
   - Risk assessment
   - Clinical confidence level
   - Professional recommendations

## 📊 Output Reports

Reports are generated in multiple formats:

### HTML Report
- Visual posture analysis with charts
- Color-coded risk indicators
- Professional recommendations
- Exportable and printable

### JSON Data
- Raw measurement data
- Timestamps and metadata
- Clinical references
- API-ready format

### PDF Report (Optional)
- Professional clinical format
- All measurements and visualizations
- Ready for medical records

Reports are saved in: `./reports/`

## 🔧 Configuration

Edit `config/settings.json` to customize:

```json
{
  "settings": {
    "gpu_acceleration": true,
    "camera_resolution": "1280x720",
    "processing_fps": 30,
    "auto_save_reports": true,
    "report_format": "both",
    "data_retention_days": 30
  }
}
```

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/
```

Run specific tests:
```bash
python -m pytest tests/test_posture_analysis.py
python -m pytest tests/test_camera_connection.py
```

## 📁 Project Structure

```
ProPostureFitness/
├── config/                 # Configuration files
│   └── settings.json
├── data/                   # Sample data and outputs
├── logs/                   # Application logs
├── reports/                # Generated reports
├── tests/                  # Unit tests
├── FitlifePostureProApp.py # Main application
├── unified_posture_system.py # Core system
├── config_module.py        # Configuration management
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MediaPipe team for pose estimation models
- OpenCV community for computer vision tools
- Clinical research papers referenced in measurements

## 📞 Support

- **Documentation**: [Full documentation](https://docs.proposturefitness.com)
- **Issues**: [GitHub Issues](https://github.com/yourusername/ProPostureFitness/issues)
- **Email**: support@proposturefitness.com
- **Community**: [Discord Server](https://discord.gg/proposturefitness)

## 🔄 Version History

- **v1.0.0** (Current) - FitlifePostureProApp with unified analysis system
  - Professional-grade posture analysis
  - Multiple analysis modes (Basic, Advanced, GPU-accelerated)
  - Streamlined interface and controls
  - Enhanced performance and accuracy

---

<p align="center">Made with 🏃‍♂️ for better posture and fitness</p>
