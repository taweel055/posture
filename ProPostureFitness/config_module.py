#!/usr/bin/env python3
"""
Configuration module for ProPostureFitness
Provides centralized path management and settings
"""

import os
import json
from pathlib import Path

class Config:
    """Central configuration for ProPostureFitness"""
    
    def __init__(self):
        # Get base directory (where this file is located)
        self.BASE_DIR = Path(__file__).parent.absolute()
        
        # Define subdirectories
        self.CONFIG_DIR = self.BASE_DIR / "config"
        self.DATA_DIR = self.BASE_DIR / "data"
        self.LOGS_DIR = self.BASE_DIR / "logs"
        self.MODELS_DIR = self.BASE_DIR / "models"
        self.REPORTS_DIR = self.BASE_DIR / "reports"
        self.BIN_DIR = self.BASE_DIR / "bin"
        
        # Create directories if they don't exist
        for dir_path in [self.DATA_DIR, self.LOGS_DIR, self.MODELS_DIR, self.REPORTS_DIR]:
            dir_path.mkdir(exist_ok=True)
        
        # Load settings
        self.settings_file = self.CONFIG_DIR / "settings.json"
        self.settings = self.load_settings()
    
    def load_settings(self):
        """Load settings from JSON file"""
        if self.settings_file.exists():
            with open(self.settings_file, 'r') as f:
                return json.load(f)
        else:
            # Return default settings if file doesn't exist
            return {
                "installation": {
                    "version": "5.0.0",
                    "install_path": str(self.BASE_DIR)
                },
                "settings": {
                    "gpu_acceleration": True,
                    "camera_resolution": "1280x720",
                    "processing_fps": 30,
                    "auto_save_reports": True,
                    "report_format": "both",
                    "data_retention_days": 30
                },
                "paths": {
                    "models": str(self.MODELS_DIR),
                    "reports": str(self.REPORTS_DIR),
                    "logs": str(self.LOGS_DIR),
                    "data": str(self.DATA_DIR)
                }
            }
    
    def save_settings(self):
        """Save current settings to JSON file"""
        with open(self.settings_file, 'w') as f:
            json.dump(self.settings, f, indent=2)
    
    def get_path(self, path_type):
        """Get a specific path from configuration"""
        path_map = {
            'base': self.BASE_DIR,
            'config': self.CONFIG_DIR,
            'data': self.DATA_DIR,
            'logs': self.LOGS_DIR,
            'models': self.MODELS_DIR,
            'reports': self.REPORTS_DIR,
            'bin': self.BIN_DIR
        }
        return path_map.get(path_type, self.BASE_DIR)
    
    @property
    def binary_path(self):
        """Get path to the compiled binary"""
        return self.BIN_DIR / "Professional_Posture_Analysis"

# Create global config instance
config = Config()

# Export commonly used paths
BASE_DIR = config.BASE_DIR
DATA_DIR = config.DATA_DIR
LOGS_DIR = config.LOGS_DIR
MODELS_DIR = config.MODELS_DIR
REPORTS_DIR = config.REPORTS_DIR
