# import uvicorn #ASGI
from fastapi import FastAPI, HTTPException
import joblib
import sklearn
import pandas as pd

def load_model():
    try:
        file = open("churn_model.pkl", "rb")
        model = joblib.load(file)
        return model
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500,detail="load model error")

def make_pred(data):
    try:
        model =load_model()
        df = pd.DataFrame([data])
        pred = model.predict(df)
        result = int(pred[0])
        return result
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500,detail="make pred")

app = FastAPI()

from pydantic import BaseModel
from datetime import date


class CustomerData(BaseModel):

    #customer_id: int
    telecom_partner: int
    gender: int
    age: int
    state: int
    city: str
    pincode: int
    #date_of_registration: date
    num_dependents: int
    estimated_salary: int
    calls_made: int
    sms_sent: int
    data_used: int

@app.post("/predict")
def predict(data: CustomerData):
    try:
        result = make_pred(data.dict())
        label = "churn" if result == 1 else "No churn"
        return {
            "prediction": result,
            "label": label
        }
    except Exception:
        raise