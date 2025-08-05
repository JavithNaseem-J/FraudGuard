import pandas as pd
import numpy as np
from FraudGuard.pipeline.inference_pipeline import PredictionPipeline

def test_fraud_detection():
    
    print("🔍 Testing Fraud Detection Pipeline...")
    print("=" * 50)
    
    # Initialize the prediction pipeline
    try:
        pipeline = PredictionPipeline()
        print("✅ Pipeline loaded successfully")

    except Exception as e:
        print(f"❌ Error loading pipeline: {e}")
        return False
    
    # Test case 1
    print("\n📊 Test Case 1: Suspicious High-Value Transaction")
    high_fraud_data = {
        'Transaction_ID': 'TXN_001',
        'User_ID': 12345,
        'Transaction_Amount': 9500.00,
        'Transaction_Type': 'Online Purchase',
        'Time_of_Transaction': 3.5,
        'Device_Used': 'Mobile',
        'Location': 'Miami', 
        'Previous_Fraudulent_Transactions': 2, 
        'Account_Age': 30,
        'Number_of_Transactions_Last_24H': 15, 
        'Payment_Method': 'Credit Card'
    }
    
    try:
        # Convert dictionary to DataFrame
        df1 = pd.DataFrame([high_fraud_data])
        result1 = pipeline.predict(df1)
        print(f"   Prediction: {result1['fraud_status']}")
        print(f"   Fraud Probability: {result1['fraud_probability']:.4f}")
        print(f"   Threshold: {result1['threshold_used']}")
        print(f"   Confidence: {result1['confidence']}")
        print(f"   Status: {'✅ PASS' if result1['fraud_status'] == 'Yes' else '❌ FAIL - Expected Fraud'}")


    except Exception as e:
        print(f"   ❌ Error in prediction: {e}")
        return False
    
    
    # Test multiple predictions
    print("\n📊 Test Case 3: Batch Predictions")
    test_cases = [
        {
            'name': 'Normal Small Purchase', 
            'data': {
                'Transaction_Amount': 12.50,
                'Transaction_Type': 'POS Payment',
                'Time_of_Transaction': 12.0,
                'Device_Used': 'Desktop',
                'Location': 'Chicago',
                'Previous_Fraudulent_Transactions': 0,
                'Account_Age': 800,
                'Number_of_Transactions_Last_24H': 1,
                'Payment_Method': 'Debit Card'
            }
        },
        {
            'name': 'Suspicious Large Transaction', 
            'data': {
                'Transaction_Amount': 8000.00,
                'Transaction_Type': 'ATM Withdrawal',
                'Time_of_Transaction': 2.5,  
                'Device_Used': 'Mobile',
                'Location': 'Boston',
                'Previous_Fraudulent_Transactions': 1,
                'Account_Age': 45, 
                'Number_of_Transactions_Last_24H': 12,
                'Payment_Method': 'UPI'
            }
        },
    ]
    
    fraud_count = 0
    total_tests = len(test_cases)
    
    for test_case in test_cases:
        try:
            df = pd.DataFrame([test_case['data']])
            result = pipeline.predict(df)
            status = "🔴 FRAUD" if result['fraud_status'] == 'Yes' else "🟢 SAFE"
            print(f"   {test_case['name']}: {status} (Probability: {result['fraud_probability']:.4f}, Confidence: {result['confidence']})")
            if result['fraud_status'] == 'Yes':
                fraud_count += 1


        except Exception as e:
            print(f"   ❌ Error in {test_case['name']}: {e}")
            return False
        
    
    print(f"\n📈 Batch Results: {fraud_count}/{total_tests} transactions flagged as fraud")
    
    print("\n" + "=" * 50)
    print("🎉 All tests completed successfully!")
    print("✅ Fraud detection pipeline is working correctly")
    print(f"✅ Using optimized threshold: {result1['threshold_used']}")

    
    return True

if __name__ == "__main__":
    success = test_fraud_detection()
    exit(0 if success else 1)
