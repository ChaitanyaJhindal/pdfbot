"""
Test script for HackRx 6.0 API endpoint.
Demonstrates how to call the /hackrx/run endpoint with the required format.
"""

import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = os.getenv("HACKRX_API_KEY", "hackrx_2024_secret_key")

# Sample request data (using the example from the specification)
sample_request = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?"
    ]
}

def test_hackrx_endpoint():
    """Test the HackRx endpoint."""
    
    # Headers with Bearer token
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    print("🚀 Testing HackRx 6.0 API Endpoint")
    print("=" * 50)
    print(f"API URL: {API_BASE_URL}/hackrx/run")
    print(f"Document URL: {sample_request['documents'][:60]}...")
    print(f"Number of questions: {len(sample_request['questions'])}")
    print()
    
    try:
        # Make the API call
        print("📤 Sending request...")
        response = requests.post(
            f"{API_BASE_URL}/hackrx/run",
            headers=headers,
            json=sample_request,
            timeout=300  # 5 minute timeout for processing
        )
        
        # Check if request was successful
        if response.status_code == 200:
            print("✅ Request successful!")
            print()
            
            # Parse response
            response_data = response.json()
            answers = response_data.get("answers", [])
            
            print("📝 Answers received:")
            print("-" * 30)
            
            for i, (question, answer) in enumerate(zip(sample_request["questions"], answers), 1):
                print(f"Question {i}: {question}")
                print(f"Answer {i}: {answer}")
                print("-" * 30)
                
        else:
            print(f"❌ Request failed with status code: {response.status_code}")
            print(f"Error details: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out. The document processing might take longer.")
    except requests.exceptions.ConnectionError:
        print("🔌 Connection error. Make sure the API server is running.")
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")

def test_health_endpoint():
    """Test the health check endpoint."""
    try:
        print("🏥 Testing health endpoint...")
        response = requests.get(f"{API_BASE_URL}/")
        
        if response.status_code == 200:
            print("✅ Health check passed!")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")

def test_invalid_auth():
    """Test authentication with invalid token."""
    print("\n🔒 Testing authentication...")
    
    headers = {
        "Authorization": "Bearer invalid_token",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/hackrx/run",
            headers=headers,
            json=sample_request,
            timeout=10
        )
        
        if response.status_code == 401:
            print("✅ Authentication correctly rejected invalid token")
        else:
            print(f"❌ Unexpected response for invalid auth: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Auth test error: {str(e)}")

if __name__ == "__main__":
    print("🧪 HackRx 6.0 API Test Suite")
    print("=" * 50)
    
    # Test health endpoint first
    test_health_endpoint()
    
    # Test authentication
    test_invalid_auth()
    
    # Test main endpoint
    print("\n" + "=" * 50)
    test_hackrx_endpoint()
    
    print("\n🎯 Test completed!")
