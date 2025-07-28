#!/usr/bin/env python3
"""
Binary verification script for ProPostureFitness
Tests the compiled binary functionality
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import stat

def verify_binary():
    """Verify the compiled binary"""
    print("🔍 ProPostureFitness Binary Verification")
    print("=" * 60)
    
    # Get binary path
    script_dir = Path(__file__).parent
    binary_path = script_dir / "bin" / "Professional_Posture_Analysis"
    
    print(f"📁 Binary location: {binary_path}")
    
    # Check if binary exists
    if not binary_path.exists():
        print("❌ Binary not found!")
        return False
        
    # Check file size
    size_mb = binary_path.stat().st_size / (1024 * 1024)
    print(f"📊 Binary size: {size_mb:.2f} MB")
    
    # Check permissions
    file_stat = binary_path.stat()
    permissions = stat.filemode(file_stat.st_mode)
    print(f"🔐 Permissions: {permissions}")
    
    # Check if executable
    is_executable = os.access(binary_path, os.X_OK)
    print(f"✅ Executable: {is_executable}")
    
    if not is_executable:
        print("🔧 Setting executable permissions...")
        binary_path.chmod(binary_path.stat().st_mode | stat.S_IEXEC)
        
    # Platform check
    system = platform.system()
    print(f"💻 Current platform: {system}")
    
    # Try to get binary info
    if system == "Darwin":  # macOS
        try:
            # Check binary architecture
            result = subprocess.run(['file', str(binary_path)], 
                                  capture_output=True, text=True)
            print(f"🏗️ Binary info: {result.stdout.strip()}")
            
            # Check dependencies
            print("\n📚 Checking dependencies...")
            result = subprocess.run(['otool', '-L', str(binary_path)], 
                                  capture_output=True, text=True)
            deps = result.stdout.strip().split('\n')[:5]  # First 5 dependencies
            for dep in deps:
                print(f"  - {dep.strip()}")
                
        except Exception as e:
            print(f"⚠️ Could not analyze binary: {e}")
            
    elif system == "Linux":
        try:
            # Check binary info
            result = subprocess.run(['file', str(binary_path)], 
                                  capture_output=True, text=True)
            print(f"🏗️ Binary info: {result.stdout.strip()}")
            
            # Check dependencies
            print("\n📚 Checking dependencies...")
            result = subprocess.run(['ldd', str(binary_path)], 
                                  capture_output=True, text=True)
            deps = result.stdout.strip().split('\n')[:5]
            for dep in deps:
                print(f"  - {dep.strip()}")
                
        except Exception as e:
            print(f"⚠️ Could not analyze binary: {e}")
    
    # Test execution (with timeout)
    print("\n🚀 Testing binary execution...")
    try:
        # Try to run with version flag
        result = subprocess.run([str(binary_path), '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ Binary executed successfully!")
            if result.stdout:
                print(f"📝 Output: {result.stdout.strip()}")
        else:
            print(f"❌ Binary returned error code: {result.returncode}")
            if result.stderr:
                print(f"📝 Error: {result.stderr.strip()}")
                
    except subprocess.TimeoutExpired:
        print("⏱️ Binary execution timed out (might need user interaction)")
    except Exception as e:
        print(f"❌ Could not execute binary: {e}")
        
    # Recommendations
    print("\n📋 Recommendations:")
    if size_mb > 300:
        print("  - Binary size is large. Consider optimization or compression.")
    if system == "Darwin" and "arm64" in str(result.stdout if 'result' in locals() else ""):
        print("  - Binary is optimized for Apple Silicon ✅")
    print("  - Test binary with actual camera hardware for full validation")
    print("  - Consider code signing for macOS distribution")
    
    return True

def create_test_launcher():
    """Create a test launcher script for the binary"""
    launcher_content = """#!/bin/bash
# Test launcher for ProPostureFitness binary

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BINARY="$SCRIPT_DIR/bin/Professional_Posture_Analysis"

echo "🚀 Launching ProPostureFitness Binary Test..."
echo "📁 Binary: $BINARY"

if [ ! -f "$BINARY" ]; then
    echo "❌ Binary not found!"
    exit 1
fi

# Make sure it's executable
chmod +x "$BINARY"

# Run with test flag if supported
"$BINARY" --test 2>/dev/null || "$BINARY"
"""
    
    script_path = Path(__file__).parent / "test_binary.sh"
    with open(script_path, 'w') as f:
        f.write(launcher_content)
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)
    print(f"\n✅ Created test launcher: {script_path}")

if __name__ == "__main__":
    verify_binary()
    create_test_launcher()
