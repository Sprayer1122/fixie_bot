#!/usr/bin/env python3
"""
Test script to demonstrate the Advanced FixieBot improvements
"""

import requests
import json
import time

def test_prediction(message):
    """Test a prediction and display results"""
    try:
        response = requests.post('http://localhost:5000/api/chat', 
                               json={'message': message})
        result = response.json()
        
        if result['success']:
            pred = result['prediction']
            print(f"\n🔍 Testing: '{message}'")
            print("=" * 50)
            print(f"📦 Module: {pred['module']} (Confidence: {pred['module_confidence']:.3f})")
            print(f"🏷️  Tag: {pred['tag']} (Confidence: {pred['tag_confidence']:.3f})")
            print(f"🔧 Fix: {pred['fix']} (Confidence: {pred['fix_confidence']:.3f})")
            print(f"📊 Overall Confidence: {pred['overall_confidence']:.3f}")
            print(f"⭐ Quality: {pred['prediction_quality']}")
            print(f"🔍 Similar Tickets Found: {len(pred['similar_tickets'])}")
            return True
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"❌ Network error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Advanced FixieBot - Model Testing")
    print("=" * 60)
    
    # Test cases that should show improved accuracy
    test_cases = [
        "Database connection errors",
        "Login button not responding", 
        "Slow loading times in dashboard",
        "Request for dark mode theme",
        "Application crashes on startup",
        "User authentication failed",
        "Dashboard interface is broken",
        "Performance issues with data loading"
    ]
    
    print("\n🎯 Testing Advanced Ensemble Model...")
    print("The model now uses 5 algorithms: Random Forest, Gradient Boosting, Logistic Regression, SVM, and Naive Bayes")
    
    success_count = 0
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}/{len(test_cases)}")
        if test_prediction(test_case):
            success_count += 1
        time.sleep(1)  # Small delay between requests
    
    print(f"\n✅ Test Results: {success_count}/{len(test_cases)} successful predictions")
    print("\n🎉 Advanced Model Features:")
    print("• Ensemble of 5 ML algorithms for robust predictions")
    print("• Advanced text preprocessing with domain-specific markers")
    print("• Intelligent confidence boosting based on keywords")
    print("• Multi-metric similarity search")
    print("• Quality assessment for each prediction")
    print("• Cross-validation for model validation")

if __name__ == "__main__":
    main() 