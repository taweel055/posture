#!/usr/bin/env python3
"""
Basic Posture Analysis System - Legacy compatibility wrapper
"""

def main():
    """Main function - now uses unified system"""
    try:
        from unified_posture_system import PostureAnalysisSystem, AnalysisMode
        print("🔄 Using unified posture analysis system (Basic Mode)...")
        system = PostureAnalysisSystem(mode=AnalysisMode.BASIC)
        system.run()
    except ImportError as e:
        print(f"❌ Could not import unified system: {e}")
        print("⚠️ Please ensure all dependencies are installed")
        import sys
        sys.exit(1)

if __name__ == "__main__":
    main()
