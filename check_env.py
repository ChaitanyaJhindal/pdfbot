"""
Diagnostic script to test if environment variables are properly loaded
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_env_vars():
    """Check if all required environment variables are set"""
    required_vars = [
        "OPENAI_API_KEY",
        "PINECONE_API_KEY", 
        "PINECONE_ENVIRONMENT",
        "HACKRX_API_KEY"
    ]
    
    print("🔍 Environment Variable Check:")
    print("=" * 40)
    
    all_good = True
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Show first/last few chars to verify without exposing full key
            masked = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else value
            print(f"✅ {var}: {masked}")
            
            # Check for common issues
            if '\n' in value:
                print(f"   ⚠️  WARNING: Contains newline character!")
                all_good = False
            if value.startswith(' ') or value.endswith(' '):
                print(f"   ⚠️  WARNING: Contains leading/trailing spaces!")
                all_good = False
        else:
            print(f"❌ {var}: NOT SET")
            all_good = False
    
    print("=" * 40)
    if all_good:
        print("✅ All environment variables look good!")
    else:
        print("❌ Some environment variables have issues!")
    
    return all_good

if __name__ == "__main__":
    check_env_vars()
