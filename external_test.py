#!/usr/bin/env python3
"""
External Testing Script for PDF Chatbot API

This script can be used by external testers to validate the API functionality.
Usage: python external_test.py <api_base_url>
Example: python external_test.py https://your-service.onrender.com
"""

import sys
import requests
import json
import time
from typing import Dict, List

class PDFChatbotTester:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'PDFChatbot-ExternalTester/1.0'
        })
    
    def test_health(self) -> bool:
        """Test the health endpoint"""
        print("🔍 Testing health endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/", timeout=30)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data.get('message', 'OK')}")
                return True
            else:
                print(f"❌ Health check failed: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {str(e)}")
            return False
    
    def test_docs_accessible(self) -> bool:
        """Test if API documentation is accessible"""
        print("📚 Testing API documentation accessibility...")
        try:
            response = self.session.get(f"{self.base_url}/docs", timeout=30)
            if response.status_code == 200:
                print("✅ API documentation is accessible")
                return True
            else:
                print(f"❌ API docs not accessible: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ API docs test error: {str(e)}")
            return False
    
    def test_api_endpoint(self, test_cases: List[Dict]) -> bool:
        """Test the main API endpoint with various test cases"""
        print("🧪 Testing main API endpoint...")
        
        all_passed = True
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i}: {test_case['name']} ---")
            
            try:
                # Add small delay to avoid overwhelming the API
                if i > 1:
                    time.sleep(2)
                
                response = self.session.post(
                    f"{self.base_url}/hackrx/run",
                    json=test_case['payload'],
                    timeout=120  # Extended timeout for PDF processing
                )
                
                if response.status_code == test_case.get('expected_status', 200):
                    if response.status_code == 200:
                        data = response.json()
                        answers = data.get('answers', [])
                        questions = test_case['payload'].get('questions', [])
                        
                        if len(answers) == len(questions):
                            print(f"✅ Test passed: Received {len(answers)} answers")
                            for j, (q, a) in enumerate(zip(questions, answers)):
                                print(f"   Q{j+1}: {q}")
                                print(f"   A{j+1}: {a[:100]}{'...' if len(a) > 100 else ''}")
                        else:
                            print(f"⚠️  Warning: Expected {len(questions)} answers, got {len(answers)}")
                    else:
                        print(f"✅ Test passed: Expected error status {response.status_code}")
                        print(f"   Error: {response.text}")
                else:
                    print(f"❌ Test failed: Expected {test_case.get('expected_status', 200)}, got {response.status_code}")
                    print(f"   Response: {response.text}")
                    all_passed = False
                    
            except requests.exceptions.Timeout:
                print("⏰ Test timed out - this might be normal for free tier cold starts")
                print("   Try again in a few minutes when the service is warm")
                all_passed = False
            except Exception as e:
                print(f"❌ Test error: {str(e)}")
                all_passed = False
        
        return all_passed
    
    def run_comprehensive_test(self) -> bool:
        """Run a comprehensive test suite"""
        print(f"🚀 Starting comprehensive test for: {self.base_url}")
        print("=" * 60)
        
        # Test cases
        test_cases = [
            {
                "name": "Simple PDF with basic questions",
                "payload": {
                    "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
                    "questions": [
                        "What is this document about?",
                        "What is the main content?"
                    ]
                },
                "expected_status": 200
            },
            {
                "name": "Single question test",
                "payload": {
                    "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
                    "questions": ["What type of document is this?"]
                },
                "expected_status": 200
            },
            {
                "name": "Invalid URL test",
                "payload": {
                    "documents": "https://invalid-url-that-does-not-exist.com/fake.pdf",
                    "questions": ["What is this about?"]
                },
                "expected_status": 400
            },
            {
                "name": "Empty questions test",
                "payload": {
                    "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
                    "questions": []
                },
                "expected_status": 400
            }
        ]
        
        # Run tests
        tests_passed = 0
        total_tests = 3  # health + docs + api
        
        # Test 1: Health check
        if self.test_health():
            tests_passed += 1
        
        print()
        
        # Test 2: Documentation
        if self.test_docs_accessible():
            tests_passed += 1
        
        print()
        
        # Test 3: API functionality
        if self.test_api_endpoint(test_cases):
            tests_passed += 1
        
        # Results
        print("\n" + "=" * 60)
        print(f"🏁 Test Results: {tests_passed}/{total_tests} tests passed")
        
        if tests_passed == total_tests:
            print("🎉 All tests passed! API is ready for external testing.")
            return True
        else:
            print("⚠️  Some tests failed. Check the issues above.")
            return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python external_test.py <api_base_url>")
        print("Example: python external_test.py https://your-service.onrender.com")
        sys.exit(1)
    
    base_url = sys.argv[1]
    tester = PDFChatbotTester(base_url)
    
    try:
        success = tester.run_comprehensive_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
