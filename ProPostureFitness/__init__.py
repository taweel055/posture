"""
ProPostureFitness Package
========================

Professional posture analysis and fitness assessment system
"""

__version__ = "5.0.0"
__author__ = "ProPostureFitness Team"
__email__ = "support@proposturefitness.com"
__license__ = "MIT"

# Import main components
from .config_module import config, BASE_DIR, DATA_DIR, LOGS_DIR, MODELS_DIR, REPORTS_DIR

# Define what gets imported with "from proposturefitness import *"
__all__ = [
    'config',
    'BASE_DIR',
    'DATA_DIR', 
    'LOGS_DIR',
    'MODELS_DIR',
    'REPORTS_DIR',
    'FinalPostureApp',
    'PostureAnalyzer'
]

# Main application class
try:
    from .final_working_app import FinalPostureApp
except ImportError:
    FinalPostureApp = None

# Additional components
try:
    from .basic_posture_app import PostureAnalyzer
except ImportError:
    PostureAnalyzer = None

def get_version():
    """Return the current version"""
    return __version__

def get_config():
    """Return the global configuration object"""
    return config
