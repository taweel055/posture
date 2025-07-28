#!/usr/bin/env python3
"""
Setup script for ProPostureFitness
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f 
                       if line.strip() and not line.startswith('#')
                       and not line.startswith('-e')]

setup(
    name="proposturefitness",
    version="5.0.0",
    author="ProPostureFitness Team",
    author_email="support@proposturefitness.com",
    description="Professional posture analysis and fitness assessment system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ProPostureFitness",
    packages=find_packages(exclude=["tests", "tests.*", "bin"]),
    include_package_data=True,
    package_data={
        'proposturefitness': [
            'config/*.json',
            'data/*.json',
            'data/*.html',
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.4.0',
        ],
        'gpu': [
            'torch>=2.0.0',
            'torchvision>=0.15.0',
            'cupy>=12.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'proposturefitness=proposturefitness.final_working_app:main',
            'pf-analyze=proposturefitness.final_working_app:main',
            'pf-gpu=proposturefitness.gpu_accelerated_posture_system:main',
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/ProPostureFitness/issues",
        "Source": "https://github.com/yourusername/ProPostureFitness",
        "Documentation": "https://docs.proposturefitness.com",
    },
)
