from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "model.pkl"
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Restaurant Revenue Prediction API",
    description="Multiple Linear Regression API for restaurant revenue prediction.",
    version="1.0.0",
)

class PredictionInput(BaseModel):
    No_Of_Item: int = Field(..., ge=1, le=200, description="Number of items ordered/handled")
    Order_Placed: int = Field(..., ge=1, le=1000, description="Number of orders placed")
    Id: int = Field(..., ge=1, le=5000, description="Restaurant ID")

@app.get("/")
def root():
    return {"message": "Restaurant Revenue Prediction API is running."}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: PredictionInput):
    try:
        X = pd.DataFrame([data.model_dump()], columns=["No_Of_Item", "Order_Placed", "Id"])
        prediction = float(model.predict(X)[0])
        return {
            "predicted_revenue": round(prediction, 2),
            "currency": "INR",
            "model": "Multiple Linear Regression"
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
