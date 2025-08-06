# Simple test with just one question
import requests
import json

# Test data with just one question
test_data = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?"
    ]
}

def test_single_question():
    url = "https://pdfbot-dkc0p0tqa-chaitanyavedansh-gmailcoms-projects.vercel.app/hackrx/run"
    headers = {
        "Authorization": "Bearer hackrx_2024_secret_key",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    print("Testing with single question...")
    print(f"URL: {url}")
    print(f"Question: {test_data['questions'][0]}")
    
    try:
        response = requests.post(url, headers=headers, json=test_data, timeout=120)  # Increased timeout
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Success! Response:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ Error Response:")
            try:
                print(json.dumps(response.json(), indent=2))
            except:
                print(response.text)
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_single_question()
