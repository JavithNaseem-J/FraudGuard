import pandas as pd

from project.pipeline.predictionpipeline import PredictionPipeline


def test_prediction():
    input_data = pd.DataFrame({
        'Transaction_Type':["ATM Withdrawal"],
        'Device_Used': ['Tablet'],
        'Location': ['Chicago'],
        'Payment_Method': ['Debit Card'],
        'Transaction_Amount':[1500000],
        'Time_of_Transaction':[16.0],
        'Previous_Fraudulent_Transactions':[4],
        'Account_Age':[119],
        'Number_of_Transactions_Last_24H':[13]
    })

    pipeline = PredictionPipeline()
    
    try:
        result = pipeline.predict(input_data)
        print("\n=== PREDICTION RESULT ===")
        print(f"Prediction: {result['fraud_status']}")
        print(f"Probability: {result['fraud_probability']:.2f}")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")

if __name__ == "__main__":
    test_prediction()