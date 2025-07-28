#!/usr/bin/env python3
"""
ProPostureFitness Quick Setup Script
Run this after cloning to set up the development environment
"""

import os
import sys
import subprocess
from pathlib import Path
import platform

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - Success")
            if result.stdout:
                print(f"   {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {description} - Failed")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} - Exception: {e}")
        return False

def main():
    """Main setup process"""
    print("🚀 ProPostureFitness v5.0 - Quick Setup")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        sys.exit(1)
    
    # Get project directory
    project_dir = Path(__file__).parent.absolute()
    os.chdir(project_dir)
    print(f"📁 Project directory: {project_dir}")
    
    # Platform info
    system = platform.system()
    print(f"💻 Platform: {system}")
    
    # Step 1: Create virtual environment
    venv_path = project_dir / "venv"
    if not venv_path.exists():
        run_command(f"{sys.executable} -m venv venv", "Creating virtual environment")
    else:
        print("✅ Virtual environment already exists")
    
    # Determine pip path
    if system == "Windows":
        pip_path = venv_path / "Scripts" / "pip"
        python_path = venv_path / "Scripts" / "python"
    else:
        pip_path = venv_path / "bin" / "pip"
        python_path = venv_path / "bin" / "python"
    
    # Step 2: Upgrade pip
    run_command(f"{pip_path} install --upgrade pip", "Upgrading pip")
    
    # Step 3: Install requirements
    if (project_dir / "requirements.txt").exists():
        run_command(f"{pip_path} install -r requirements.txt", "Installing dependencies")
    else:
        print("⚠️ requirements.txt not found")
    
    # Step 4: Install package in development mode
    run_command(f"{pip_path} install -e .", "Installing ProPostureFitness package")
    
    # Step 5: Create necessary directories
    directories = ["logs", "reports", "models", "data/temp"]
    for dir_name in directories:
        dir_path = project_dir / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
    print("✅ Created necessary directories")
    
    # Step 6: Make scripts executable
    if system != "Windows":
        scripts = ["ProPostureFitness.sh", "tests/run_tests.py", "verify_binary.py"]
        for script in scripts:
            script_path = project_dir / script
            if script_path.exists():
                run_command(f"chmod +x {script_path}", f"Making {script} executable")
    
    # Step 7: Verify installation
    print("\n🔍 Verifying installation...")
    test_imports = [
        "cv2",
        "numpy",
        "mediapipe"
    ]
    
    all_good = True
    for module in test_imports:
        try:
            subprocess.run([str(python_path), "-c", f"import {module}"], 
                         check=True, capture_output=True)
            print(f"✅ {module} - OK")
        except:
            print(f"❌ {module} - Failed to import")
            all_good = False
    
    # Final summary
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Activate virtual environment:")
        if system == "Windows":
            print("   .\\venv\\Scripts\\activate")
        else:
            print("   source venv/bin/activate")
        print("2. Run tests:")
        print("   python tests/run_tests.py")
        print("3. Start the application:")
        print("   python final_working_app.py")
        print("\n💡 Or use the Makefile:")
        print("   make test")
        print("   make run")
    else:
        print("⚠️ Setup completed with warnings")
        print("Some dependencies may need manual installation")
    
    print("\n📚 See README.md for full documentation")
    print("🐛 Report issues at: https://github.com/yourusername/ProPostureFitness/issues")

if __name__ == "__main__":
    main()
