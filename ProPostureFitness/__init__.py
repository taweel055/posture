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
from .unified_posture_system import PostureAnalysisSystem, AnalysisMode

# Define what gets imported with "from proposturefitness import *"
__all__ = [
    'config',
    'BASE_DIR',
    'DATA_DIR', 
    'LOGS_DIR',
    'MODELS_DIR',
    'REPORTS_DIR',
    'PostureAnalysisSystem',
    'AnalysisMode'
]


def get_version():
    """Return the current version"""
    return __version__

def get_config():
    """Return the global configuration object"""
    return config
