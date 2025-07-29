#!/usr/bin/env python3
"""
Startup script for the AI Job Offer Detector
"""

import subprocess
import sys
import os
import time
import webbrowser
import threading

def install_requirements():
    """Install required Python packages"""
    print("📦 Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def start_backend():
    """Start the Flask backend server"""
    print("🚀 Starting AI Job Offer Detector Backend...")
    print("📍 Backend will be available at: http://localhost:5000")
    print("🔍 API Endpoints:")
    print("   - POST /predict - Analyze job offer text")
    print("   - POST /upload - Upload and analyze file")
    print("   - GET /health - Health check")
    print("   - GET /model-info - Model information")
    print("\n" + "="*60)
    
    try:
        # Start the Flask app
        subprocess.run([sys.executable, "backend.py"])
    except KeyboardInterrupt:
        print("\n🛑 Backend server stopped by user")
    except Exception as e:
        print(f"❌ Error starting backend: {e}")

def start_frontend():
    """Start the React frontend"""
    print("🌐 Starting React Frontend...")
    print("📍 Frontend will be available at: http://localhost:3000")
    
    try:
        # Change to frontend directory
        os.chdir('frontend')
        
        # Install dependencies if needed
        if not os.path.exists('node_modules'):
            print("📦 Installing frontend dependencies...")
            subprocess.run(['npm', 'install'], check=True)
        
        # Start the React app
        subprocess.run(['npm', 'start'], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Frontend server stopped by user")
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")
    finally:
        # Change back to root directory
        os.chdir('..')

def open_browser():
    """Open browser after a delay"""
    time.sleep(5)  # Wait for servers to start
    try:
        webbrowser.open('http://localhost:3000')
        print("🌐 Opening application in browser...")
    except:
        print("💡 Open http://localhost:3000 in your browser to access the application")

def main():
    """Main startup function"""
    print("🛡️ AI Job Offer Detector - Startup")
    print("=" * 60)
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        sys.exit(1)
    
    # Install dependencies
    if not install_requirements():
        sys.exit(1)
    
    # Check if frontend directory exists
    if not os.path.exists("frontend"):
        print("❌ frontend directory not found!")
        sys.exit(1)
    
    print("\n🎯 Starting both backend and frontend servers...")
    print("💡 Press Ctrl+C to stop both servers")
    print("=" * 60)
    
    # Start browser in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=start_backend)
    backend_thread.daemon = True
    backend_thread.start()
    
    # Start frontend in main thread
    try:
        start_frontend()
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")

if __name__ == "__main__":
    main() 