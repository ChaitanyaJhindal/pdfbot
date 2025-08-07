"""
Test script for the deployed PDF Chatbot API on Render
"""

import requests
import json
import sys

def test_api(base_url):
    """Test the deployed API endpoints"""
    
    print(f"Testing API at: {base_url}")
    print("=" * 50)
    
    # Test 1: Health check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    print()
    
    # Test 2: API endpoint with sample data
    print("2. Testing HackRx endpoint...")
    
    # Sample test data
    test_data = {
        "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        "questions": [
            "What is this document about?",
            "What is the main content?"
        ]
    }
    
    try:
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            f"{base_url}/hackrx/run",
            json=test_data,
            headers=headers,
            timeout=120  # Extended timeout for PDF processing
        )
        
        if response.status_code == 200:
            print("✅ API endpoint test passed")
            result = response.json()
            print(f"   Questions: {len(test_data['questions'])}")
            print(f"   Answers received: {len(result.get('answers', []))}")
            for i, answer in enumerate(result.get('answers', [])):
                print(f"   Answer {i+1}: {answer[:100]}...")
        else:
            print(f"❌ API endpoint test failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ API request timed out (this might be normal for free tier)")
        print("   Try again or consider upgrading to a paid plan")
        return False
    except Exception as e:
        print(f"❌ API endpoint error: {e}")
        return False
    
    print()
    print("🎉 All tests completed successfully!")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_render_api.py <base_url>")
        print("Example: python test_render_api.py https://your-service-name.onrender.com")
        sys.exit(1)
    
    base_url = sys.argv[1].rstrip('/')
    test_api(base_url)
