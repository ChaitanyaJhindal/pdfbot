# Test script for HackRx API compliance
import requests
import json

# Test data from HackRx specification
test_data = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}

# Test locally first
def test_local_api():
    url = "http://localhost:8000/hackrx/run"
    headers = {
        "Authorization": "Bearer hackrx_2024_secret_key",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    try:
        response = requests.post(url, headers=headers, json=test_data, timeout=60)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

# Test Vercel deployment
def test_vercel_api(vercel_url, test_payload=None):
    url = f"{vercel_url}/hackrx/run"
    headers = {
        "Authorization": "Bearer hackrx_2024_secret_key",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    # Use provided test payload or default test_data
    payload = test_payload if test_payload else test_data
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing HackRx API compliance...")
    print("=" * 50)
    
    # Test with smaller subset first
    small_test = {
        "documents": test_data["documents"],
        "questions": test_data["questions"][:2]  # Only first 2 questions
    }
    
    print("Testing with 2 questions first...")
    
    # Test the actual Vercel deployment
    vercel_url = "https://pdfbot-5oxs-4y0puw2oy-chaitanyavedansh-gmailcoms-projects.vercel.app"
    print(f"\nTesting Vercel deployment: {vercel_url}")
    success = test_vercel_api(vercel_url, small_test)  # Use the small_test payload
    
    if success:
        print("\n✅ HackRx API is working correctly!")
    else:
        print("\n❌ HackRx API test failed. Check the logs above for details.")
