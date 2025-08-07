#!/usr/bin/env python3
"""
HackRx 6.0 Competition Setup Script

This script helps set up and validate your environment for the competition.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print("❌ Python 3.8+ required. Current version:", f"{version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_requirements():
    """Check if requirements.txt exists."""
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    print("✅ requirements.txt found")
    return True

def install_dependencies():
    """Install dependencies from requirements.txt."""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e.stderr}")
        return False

def check_environment_variables():
    """Check if required environment variables are set."""
    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
    optional_vars = ["PINECONE_ENVIRONMENT", "HACKRX_API_KEY", "LOG_LEVEL"]
    
    print("🔍 Checking environment variables...")
    
    all_good = True
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var}: Set")
        else:
            print(f"❌ {var}: Not set (REQUIRED)")
            all_good = False
    
    for var in optional_vars:
        if os.getenv(var):
            print(f"✅ {var}: Set")
        else:
            print(f"⚪ {var}: Not set (optional)")
    
    return all_good

def create_env_template():
    """Create .env template file."""
    env_content = """# HackRx 6.0 Competition Environment Variables

# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here

# Optional Configuration
PINECONE_ENVIRONMENT=gcp-starter
HACKRX_API_KEY=hackrx_2024_secret_key
LOG_LEVEL=INFO
ENVIRONMENT=hackrx

# For local development
PORT=8000
"""
    
    if not Path(".env").exists():
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ Created .env template file")
        print("📝 Please edit .env file with your actual API keys")
    else:
        print("⚪ .env file already exists")

def test_local_startup():
    """Test if the API can start locally."""
    print("🧪 Testing local API startup...")
    try:
        # Import test
        import fastapi
        import uvicorn
        import pydantic
        print("✅ Core dependencies import successfully")
        
        # Try to import our app
        sys.path.append('.')
        from api import app
        print("✅ API application imports successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Startup error: {e}")
        return False

def show_deployment_instructions():
    """Show deployment instructions."""
    print("\n" + "="*60)
    print("🚀 DEPLOYMENT INSTRUCTIONS")
    print("="*60)
    print()
    print("1. Push your code to GitHub:")
    print("   git add .")
    print("   git commit -m 'HackRx 6.0 competition ready'")
    print("   git push origin main")
    print()
    print("2. Deploy to Render:")
    print("   - Go to https://render.com")
    print("   - Click 'New +' → 'Web Service'")
    print("   - Connect your GitHub repository")
    print("   - Use the settings from render.yaml")
    print()
    print("3. Set environment variables in Render:")
    print("   ENVIRONMENT=hackrx")
    print("   OPENAI_API_KEY=your_actual_key")
    print("   PINECONE_API_KEY=your_actual_key")
    print("   HACKRX_API_KEY=hackrx_2024_secret_key")
    print()
    print("4. Test your deployment:")
    print("   python hackrx_test.py https://your-service.onrender.com")
    print()
    print("📚 See HACKRX_COMPETITION_GUIDE.md for detailed instructions")

def main():
    print("🎮 HackRx 6.0 Competition Setup Script")
    print("="*50)
    
    checks_passed = 0
    total_checks = 5
    
    # Check Python version
    if check_python_version():
        checks_passed += 1
    
    # Check requirements.txt
    if check_requirements():
        checks_passed += 1
    
    # Install dependencies
    if install_dependencies():
        checks_passed += 1
    
    # Check environment variables
    if check_environment_variables():
        checks_passed += 1
    else:
        create_env_template()
    
    # Test local startup
    if test_local_startup():
        checks_passed += 1
    
    print("\n" + "="*50)
    print(f"✅ Setup Results: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed == total_checks:
        print("🎉 Your environment is ready for HackRx 6.0 competition!")
        show_deployment_instructions()
    elif checks_passed >= total_checks - 1:
        print("⚠️  Almost ready! Please fix the remaining issues.")
        if not os.getenv("OPENAI_API_KEY") or not os.getenv("PINECONE_API_KEY"):
            print("🔑 Don't forget to set your API keys in the .env file!")
    else:
        print("❌ Several issues need to be fixed before deployment.")
    
    print("\n📋 Next steps:")
    print("1. Fix any issues above")
    print("2. Set your API keys in .env file")
    print("3. Test locally: uvicorn api:app --reload")
    print("4. Deploy to Render following the guide")
    print("5. Test with: python hackrx_test.py <your-deployed-url>")

if __name__ == "__main__":
    main()
