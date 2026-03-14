from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

MODEL_PATH = "models/best.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

class SalesFeatures(BaseModel):

    Store: int
    Temperature: float
    Fuel_Price: float
    CPI: float
    Unemployment: float
    Holiday_Flag: int

    lag_1: float
    lag_2: float
    lag_4: float

    rolling_mean_4: float
    rolling_std_4: float

    day_of_week: int
    month: int
    week_of_year: int
    is_weekend: int

@app.post("/predict")
def predict(features: SalesFeatures):

    data = pd.DataFrame([features.dict()])

    prediction = model.predict(data)[0]

    return {
        "predicted_weekly_sales": float(prediction)
    }