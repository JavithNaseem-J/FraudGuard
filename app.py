from flask import Flask, request, render_template, jsonify, redirect, url_for
import os
import pandas as pd
import json
import traceback
from FraudGuard.pipeline.predictionpipeline import PredictionPipeline

app = Flask(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, 'templates')

app.template_folder = template_dir

@app.route("/")
def home_page():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        transaction_type = request.form.get('Transaction_Type')
        device_used = request.form.get('Device_Used')
        location = request.form.get('Location')
        payment_method = request.form.get('Payment_Method')
        transaction_amount = float(request.form.get('Transaction_Amount'))
        time_of_transaction = float(request.form.get('Time_of_Transaction'))
        previous_fraudulent_transactions = int(request.form.get('Previous_Fraudulent_Transactions'))
        account_age = int(request.form.get('Account_Age'))
        number_of_transactions_last_24h = int(request.form.get('Number_of_Transactions_Last_24H'))

        data = {
            'Transaction_Type':[request.form['Transaction_Type']],
            'Device_Used':[request.form['Device_Used']],
            'Location':[request.form['Location']],
            'Payment_Method':[request.form['Payment_Method']],
            'Transaction_Amount':[float(request.form['Transaction_Amount'])],
            'Time_of_Transaction':[float(request.form['Time_of_Transaction'])],
            'Previous_Fraudulent_Transactions':[int(request.form['Previous_Fraudulent_Transactions'])],
            'Account_Age':[int(request.form['Account_Age'])],
            'Number_of_Transactions_Last_24H':[int(request.form['Number_of_Transactions_Last_24H'])]
        }

        df = pd.DataFrame(data)


        pipeline = PredictionPipeline()
        result = pipeline.predict(df)
        
        fraud_status = result['fraud_status']
        fraud_probability = result['fraud_probability']
        
        print(f"Prediction result: {fraud_status}, Probability: {fraud_probability:.2f}")

        encoded_data = json.dumps(data)
        
        return jsonify({
            "redirect": f"/results?fraud_status={fraud_status}&fraud_probability={fraud_probability}&data={encoded_data}"
        })

    except Exception as e:
        print(f"Error in prediction endpoint: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

@app.route("/results")
def show_results():
    try:
        fraud_status = request.args.get('fraud_status')
        fraud_probability = request.args.get('fraud_probability')
        data = request.args.get('data')
        
        if fraud_status is None or fraud_probability is None or data is None:
            print("Missing required parameters for results page")
            return redirect(url_for('home_page'))

        print(f"Received fraud_status: {fraud_status}")
        print(f"Received fraud_probability: {fraud_probability}")
        print(f"Received data: {data}")
        
        input_data = json.loads(data)
        
        print(f"Parsed input_data: {input_data}")
        
        return render_template("result.html",
            fraud_status=fraud_status,
            fraud_probability=float(fraud_probability),
            transaction_data=input_data
        )
    except Exception as e:
        print(f"Error in results handler: {str(e)}")
        print(traceback.format_exc())
        return redirect(url_for('home_page'))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)