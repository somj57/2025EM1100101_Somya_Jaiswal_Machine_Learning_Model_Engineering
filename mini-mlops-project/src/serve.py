from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from features import engineer_features

app = FastAPI(title="Churn Prediction API")

# Load model into memory on startup
try:
    model = joblib.load('models/promoted_model.pkl')
except FileNotFoundError:
    model = None

# Input schema
class CustomerData(BaseModel):
    customerID: str
    tenure: int
    MonthlyCharges: float
    SeniorCitizen: int
    InternetService: str
    PhoneService: str
    Contract: str

@app.post("/predict")
def predict(data: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model artifact missing.")
    
    # Map JSON payload to dataframe
    df_in = pd.DataFrame([data.model_dump()])
    
    # Call the exact same preprocessing function used in train.py
    df_features = engineer_features(df_in)
    
    prediction = model.predict(df_features)[0]
    probability = model.predict_proba(df_features)[0][1]
    
    return {
        "customerID": data.customerID,
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4),
        "model_version": "v2.0-candidate",
        "action": "Offer Retention Promo" if prediction == 1 else "No Action"
    }