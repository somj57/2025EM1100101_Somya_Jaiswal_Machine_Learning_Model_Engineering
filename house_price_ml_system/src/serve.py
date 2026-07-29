import sys
import os

# 1. Explicitly add the 'src' directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# 2. Import the custom transformer so joblib knows how to reconstruct it
from features import HouseFeatureEngineer

app = FastAPI(title="House Price Prediction API", version="1.0")

# Load model on startup
model = joblib.load("models/v1/model.joblib")
MODEL_VERSION = "v1.0.0-RandomForest"

# Define input schema using Field aliases
class HouseRequest(BaseModel):
    YrSold: int
    YearBuilt: int
    YearRemodAdd: int
    TotalBsmtSF: float
    FirstFlrSF: float = Field(alias="1stFlrSF")
    SecondFlrSF: float = Field(alias="2ndFlrSF")
    FullBath: int
    HalfBath: int
    BsmtFullBath: int
    BsmtHalfBath: int
    PoolArea: float
    OverallQual: int

@app.post("/predict")
def predict(request: HouseRequest):
    # Convert request to dataframe 
    df = pd.DataFrame([request.model_dump(by_alias=True)])
    
    # Predict (Feature engineering runs automatically via the pipeline)
    prediction = model.predict(df)[0]
    
    return {
        "predicted_price": round(prediction, 2),
        "model_version": MODEL_VERSION,
        "currency": "USD"
    }