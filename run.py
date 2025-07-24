#!/usr/bin/env python3
"""
FixieBot Startup Script
A simple script to run the FixieBot application with better error handling.
"""

import sys
import os
import subprocess
import time

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required!")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version.split()[0]}")

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'flask', 'pandas', 'numpy', 'scikit-learn', 
        'transformers', 'torch', 'sentence_transformers'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Installing dependencies...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies!")
            sys.exit(1)
    else:
        print("✅ All dependencies are installed!")

def check_data_files():
    """Check if required data files exist"""
    required_files = [
        'dataset/historical_tickets.csv',
        'dataset/new_tickets.csv'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing data files: {', '.join(missing_files)}")
        print("Please ensure all dataset files are present!")
        sys.exit(1)
    else:
        print("✅ All data files found!")

def main():
    """Main startup function"""
    print("🤖 FixieBot - AI-Powered Ticket Fix Predictor")
    print("=" * 50)
    
    # Check prerequisites
    print("\n🔍 Checking prerequisites...")
    check_python_version()
    check_dependencies()
    check_data_files()
    
    print("\n🚀 Starting FixieBot...")
    print("📝 This may take a few seconds to load the ML models...")
    print("🌐 The application will be available at: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Import and run the Flask app
        from app import app, fixie_bot
        
        # Check if models are ready
        if not fixie_bot.is_trained:
            print("🔄 Training ML models...")
            if fixie_bot.train_models():
                print("✅ Models trained successfully!")
            else:
                print("❌ Failed to train models!")
                sys.exit(1)
        else:
            print("✅ ML models are ready!")
        
        print("\n🎉 FixieBot is ready! Opening in your browser...")
        
        # Try to open browser automatically
        try:
            import webbrowser
            time.sleep(2)  # Give the server time to start
            webbrowser.open('http://localhost:5000')
        except:
            pass  # Browser opening is optional
        
        # Run the Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except KeyboardInterrupt:
        print("\n\n👋 FixieBot stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting FixieBot: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 