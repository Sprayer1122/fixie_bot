#!/usr/bin/env python3
"""
Comprehensive FixieBot Regression Testing Script
Tests all functionality to ensure the chatbot is working perfectly.
"""

import requests
import json
import time
from typing import Dict, Any

class FixieBotTester:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.test_results = []
        
    def test_endpoint(self, endpoint: str, method: str = "GET", data: Dict = None) -> Dict[str, Any]:
        """Test an API endpoint and return results"""
        try:
            url = f"{self.base_url}{endpoint}"
            if method == "GET":
                response = requests.get(url)
            elif method == "POST":
                response = requests.post(url, json=data, headers={'Content-Type': 'application/json'})
            
            return {
                'status_code': response.status_code,
                'success': response.status_code == 200,
                'data': response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
            }
        except Exception as e:
            return {
                'status_code': 0,
                'success': False,
                'error': str(e)
            }
    
    def test_basic_functionality(self):
        """Test basic API endpoints"""
        print("🔍 Testing Basic Functionality...")
        
        # Test 1: Frontend loads
        result = self.test_endpoint("/")
        self.test_results.append({
            'test': 'Frontend Loads',
            'success': result['success'],
            'details': f"Status: {result['status_code']}"
        })
        print(f"  ✅ Frontend: {'PASS' if result['success'] else 'FAIL'}")
        
        # Test 2: Status endpoint
        result = self.test_endpoint("/api/status")
        self.test_results.append({
            'test': 'Status Endpoint',
            'success': result['success'],
            'details': f"Status: {result['status_code']}, Models Trained: {result.get('data', {}).get('models_trained', False)}"
        })
        print(f"  ✅ Status API: {'PASS' if result['success'] else 'FAIL'}")
        
        # Test 3: Retrain endpoint
        result = self.test_endpoint("/api/force-retrain")
        self.test_results.append({
            'test': 'Retrain Endpoint',
            'success': result['success'],
            'details': f"Status: {result['status_code']}"
        })
        print(f"  ✅ Retrain API: {'PASS' if result['success'] else 'FAIL'}")
    
    def test_prediction_accuracy(self):
        """Test prediction accuracy for different types of issues"""
        print("\n🎯 Testing Prediction Accuracy...")
        
        test_cases = [
            {
                'input': 'Database connection errors',
                'expected_module': 'ModuleC',
                'expected_tag': 'Bug',
                'description': 'Database Issues'
            },
            {
                'input': 'Slow loading times in dashboard',
                'expected_module': 'ModuleA',
                'expected_tag': 'Performance',
                'description': 'UI Performance Issues'
            },
            {
                'input': 'Login button not responding',
                'expected_module': 'ModuleD',
                'expected_tag': 'UI Issue',
                'description': 'Authentication Issues'
            },
            {
                'input': 'Request for dark mode theme',
                'expected_module': 'ModuleA',
                'expected_tag': 'Feature Request',
                'description': 'Feature Requests'
            },
            {
                'input': 'Application crashes on startup',
                'expected_module': 'ModuleB',
                'expected_tag': 'Bug',
                'description': 'Crash Issues'
            }
        ]
        
        for test_case in test_cases:
            result = self.test_endpoint("/api/chat", "POST", {"message": test_case['input']})
            
            if result['success']:
                prediction = result['data']['prediction']
                module_correct = prediction['module'] == test_case['expected_module']
                tag_correct = prediction['tag'] == test_case['expected_tag']
                
                self.test_results.append({
                    'test': f"Prediction: {test_case['description']}",
                    'success': module_correct and tag_correct,
                    'details': f"Module: {prediction['module']} (expected: {test_case['expected_module']}), Tag: {prediction['tag']} (expected: {test_case['expected_tag']})"
                })
                
                status = "PASS" if module_correct and tag_correct else "FAIL"
                print(f"  {status} {test_case['description']}: Module={prediction['module']}, Tag={prediction['tag']}")
            else:
                self.test_results.append({
                    'test': f"Prediction: {test_case['description']}",
                    'success': False,
                    'details': f"API Error: {result.get('error', 'Unknown')}"
                })
                print(f"  ❌ {test_case['description']}: API Error")
    
    def test_diversity(self):
        """Test that random inputs produce diverse predictions"""
        print("\n🎲 Testing Prediction Diversity...")
        
        random_inputs = ['sdfsd', 'test', 'hello', 'random text', 'xyz123']
        predictions = []
        
        for input_text in random_inputs:
            result = self.test_endpoint("/api/chat", "POST", {"message": input_text})
            if result['success']:
                prediction = result['data']['prediction']
                predictions.append({
                    'input': input_text,
                    'module': prediction['module'],
                    'tag': prediction['tag'],
                    'fix': prediction['fix']
                })
        
        # Check for diversity
        unique_modules = len(set(p['module'] for p in predictions))
        unique_tags = len(set(p['tag'] for p in predictions))
        unique_fixes = len(set(p['fix'] for p in predictions))
        
        diversity_score = (unique_modules + unique_tags + unique_fixes) / 3
        is_diverse = diversity_score >= 2.0  # At least 2 different values on average
        
        self.test_results.append({
            'test': 'Prediction Diversity',
            'success': is_diverse,
            'details': f"Diversity Score: {diversity_score:.2f} (Modules: {unique_modules}, Tags: {unique_tags}, Fixes: {unique_fixes})"
        })
        
        print(f"  {'✅' if is_diverse else '❌'} Diversity: Score {diversity_score:.2f}")
        for pred in predictions:
            print(f"    {pred['input']}: Module={pred['module']}, Tag={pred['tag']}")
    
    def test_similar_tickets(self):
        """Test that similar tickets are returned"""
        print("\n📋 Testing Similar Tickets...")
        
        result = self.test_endpoint("/api/chat", "POST", {"message": "Database connection errors"})
        
        if result['success']:
            prediction = result['data']['prediction']
            similar_tickets = prediction.get('similar_tickets', [])
            
            has_similar_tickets = len(similar_tickets) > 0
            tickets_have_required_fields = all(
                all(key in ticket for key in ['ticket_id', 'description', 'module', 'fix', 'tag', 'similarity'])
                for ticket in similar_tickets
            )
            
            self.test_results.append({
                'test': 'Similar Tickets',
                'success': has_similar_tickets and tickets_have_required_fields,
                'details': f"Found {len(similar_tickets)} similar tickets with required fields: {tickets_have_required_fields}"
            })
            
            print(f"  {'✅' if has_similar_tickets and tickets_have_required_fields else '❌'} Similar Tickets: {len(similar_tickets)} tickets found")
        else:
            self.test_results.append({
                'test': 'Similar Tickets',
                'success': False,
                'details': 'API Error'
            })
            print("  ❌ Similar Tickets: API Error")
    
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        print("\n⚠️ Testing Error Handling...")
        
        # Test empty message
        result = self.test_endpoint("/api/chat", "POST", {"message": ""})
        self.test_results.append({
            'test': 'Empty Message Handling',
            'success': not result['success'],  # Should fail for empty message
            'details': f"Status: {result['status_code']}"
        })
        print(f"  {'✅' if not result['success'] else '❌'} Empty Message: {'Handled' if not result['success'] else 'Not Handled'}")
        
        # Test missing message field
        result = self.test_endpoint("/api/chat", "POST", {})
        self.test_results.append({
            'test': 'Missing Message Field',
            'success': not result['success'],  # Should fail for missing message
            'details': f"Status: {result['status_code']}"
        })
        print(f"  {'✅' if not result['success'] else '❌'} Missing Message: {'Handled' if not result['success'] else 'Not Handled'}")
    
    def run_all_tests(self):
        """Run all tests and generate report"""
        print("🤖 FixieBot Comprehensive Regression Testing")
        print("=" * 50)
        
        self.test_basic_functionality()
        self.test_prediction_accuracy()
        self.test_diversity()
        self.test_similar_tickets()
        self.test_error_handling()
        
        # Generate summary
        print("\n📊 Test Summary")
        print("=" * 50)
        
        passed = sum(1 for result in self.test_results if result['success'])
        total = len(self.test_results)
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if passed == total:
            print("\n🎉 ALL TESTS PASSED! FixieBot is working perfectly!")
        else:
            print("\n❌ Some tests failed. Details:")
            for result in self.test_results:
                if not result['success']:
                    print(f"  - {result['test']}: {result['details']}")
        
        return passed == total

if __name__ == "__main__":
    tester = FixieBotTester()
    success = tester.run_all_tests()
    exit(0 if success else 1) 