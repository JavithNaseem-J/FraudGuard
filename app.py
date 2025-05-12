from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import pandas as pd
import json
import uvicorn
from project.pipeline.predictionpipeline import PredictionPipeline

app = FastAPI()

current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, 'templates')
templates = Jinja2Templates(directory=template_dir)

static_dir = os.path.join(current_dir, 'static')
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/api/placeholder/{width}/{height}")
async def placeholder_image(width: int, height: int):
    return JSONResponse(content={"width": width, "height": height})

@app.get("/")
async def home_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(
    request: Request,
    Transaction_Type: str = Form(...),
    Device_Used: str = Form(...),
    Location: str = Form(...),
    Payment_Method: str = Form(...),
    Transaction_Amount: float = Form(...),
    Time_of_Transaction: float = Form(...),
    Previous_Fraudulent_Transactions: int = Form(...),
    Account_Age: int = Form(...),
    Number_of_Transactions_Last_24H: int = Form(...)
):
    try:
        data = {
            "Transaction_Type": Transaction_Type,
            "Device_Used": Device_Used,
            "Location": Location,
            "Payment_Method": Payment_Method,
            "Transaction_Amount": Transaction_Amount,
            "Time_of_Transaction": Time_of_Transaction,
            "Previous_Fraudulent_Transactions": Previous_Fraudulent_Transactions,
            "Account_Age": Account_Age,
            "Number_of_Transactions_Last_24H": Number_of_Transactions_Last_24H
        }

        input_df = pd.DataFrame({
            'Transaction_Type': [Transaction_Type],
            'Device_Used': [Device_Used],
            'Location': [Location],
            'Payment_Method': [Payment_Method],
            'Transaction_Amount': [Transaction_Amount],
            'Time_of_Transaction': [Time_of_Transaction],
            'Previous_Fraudulent_Transactions': [Previous_Fraudulent_Transactions],
            'Account_Age': [Account_Age],
            'Number_of_Transactions_Last_24H': [Number_of_Transactions_Last_24H]
        })

        print(f"Input DataFrame columns: {input_df.columns.tolist()}")
        print(f"Input DataFrame values: {input_df.iloc[0].tolist()}")

        pipeline = PredictionPipeline()
        result = pipeline.predict(input_df)
        
        fraud_status = result['fraud_status']
        fraud_probability = result['fraud_probability']
        
        print(f"Prediction result: {fraud_status}, Probability: {fraud_probability:.2f}")

        encoded_data = json.dumps(data)
        

        return JSONResponse(content={
            "redirect": f"/results?fraud_status={fraud_status}&fraud_probability={fraud_probability}&data={encoded_data}"
        })

    except Exception as e:
        import traceback
        print(f"Error in prediction endpoint: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(content={"error": f"Error during prediction: {str(e)}"}, status_code=500)
    

@app.get("/results")
async def show_results(
    request: Request, 
    fraud_status: str = None, 
    fraud_probability: float = None, 
    data: str = None
):
    try:
        if fraud_status is None or fraud_probability is None or data is None:
            print("Missing required parameters for results page")
            return RedirectResponse(url="/")

        print(f"Received fraud_status: {fraud_status}")
        print(f"Received fraud_probability: {fraud_probability}")
        print(f"Received data: {data}")
        
        input_data = json.loads(data)
        
        print(f"Parsed input_data: {input_data}")
        
        return templates.TemplateResponse("result.html", {
            "request": request,
            "fraud_status": fraud_status,
            "fraud_probability": float(fraud_probability),
            "transaction_data": input_data
        })
    except Exception as e:
        import traceback
        print(f"Error in results handler: {str(e)}")
        print(traceback.format_exc())
        return RedirectResponse(url="/")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)