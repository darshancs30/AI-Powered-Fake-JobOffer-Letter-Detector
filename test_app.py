#!/usr/bin/env python3
"""
Test script for the AI Job Offer Detector
"""

import requests
import json
import time

def test_health_endpoint():
    """Test the health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get("http://localhost:5000/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_model_info():
    """Test the model info endpoint"""
    print("\n🔍 Testing model info endpoint...")
    try:
        response = requests.get("http://localhost:5000/model-info")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info: Accuracy {data.get('accuracy', 'N/A')}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model info error: {e}")
        return False

def test_predict_endpoint():
    """Test the predict endpoint with sample data"""
    print("\n🔍 Testing predict endpoint...")
    
    # Sample real job offer
    real_offer = {
        "text": "We are pleased to offer you the position of Software Engineer at our company. This is a full-time position with competitive salary and benefits including health insurance, 401k, and paid time off. Your start date will be Monday, January 15th, 2024. We look forward to having you join our team."
    }
    
    # Sample fake job offer
    fake_offer = {
        "text": "URGENT: You have been selected for a high-paying job! Send $500 processing fee immediately to secure your position. This is a limited time offer. Call now to claim your dream job! Send your bank details and personal information."
    }
    
    try:
        # Test real offer
        print("📝 Testing with real job offer...")
        response = requests.post("http://localhost:5000/predict", json=real_offer)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Real offer prediction: {data['prediction']} (Confidence: {data['confidence_percentage']})")
        else:
            print(f"❌ Real offer test failed: {response.status_code}")
            return False
        
        # Test fake offer
        print("📝 Testing with fake job offer...")
        response = requests.post("http://localhost:5000/predict", json=fake_offer)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Fake offer prediction: {data['prediction']} (Confidence: {data['confidence_percentage']})")
        else:
            print(f"❌ Fake offer test failed: {response.status_code}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Predict test error: {e}")
        return False

def test_error_handling():
    """Test error handling"""
    print("\n🔍 Testing error handling...")
    
    # Test empty text
    try:
        response = requests.post("http://localhost:5000/predict", json={"text": ""})
        if response.status_code == 400:
            print("✅ Empty text error handling works")
        else:
            print(f"❌ Empty text error handling failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Empty text test error: {e}")
        return False
    
    # Test missing text field
    try:
        response = requests.post("http://localhost:5000/predict", json={})
        if response.status_code == 400:
            print("✅ Missing text field error handling works")
        else:
            print(f"❌ Missing text field error handling failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Missing text field test error: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🧪 AI Job Offer Detector - API Tests")
    print("=" * 50)
    
    # Wait a moment for server to be ready
    time.sleep(2)
    
    tests = [
        test_health_endpoint,
        test_model_info,
        test_predict_endpoint,
        test_error_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The API is working correctly.")
        print("\n🌐 Frontend should be available at: http://localhost:3000")
        print("🔧 Backend API is available at: http://localhost:5000")
        print("\n💡 You can now:")
        print("   - Upload files (PDF, DOCX, TXT, images)")
        print("   - Paste text for analysis")
        print("   - View detailed fraud indicators and explanations")
    else:
        print("❌ Some tests failed. Please check the server logs.")

if __name__ == "__main__":
    main() 