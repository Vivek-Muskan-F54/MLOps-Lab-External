from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "water_shortage_model.joblib"
app = FastAPI(title="Overnight Water Shortage Prediction API", version="1.0")

class EveningConditions(BaseModel):
    tank_level_8pm: float = Field(ge=0, le=100000)
    connected_households: int = Field(gt=0)
    avg_hourly_consumption: float = Field(gt=0)
    expected_pumping_duration: float = Field(ge=0, le=10)
    expected_pumping_rate: float = Field(ge=0)
    leakage_estimate: float = Field(ge=0)
    day_type: str

def make_features(data):
    if data.day_type not in ("Weekday", "Weekend"):
        raise ValueError("day_type must be Weekday or Weekend")
    demand = data.connected_households * data.avg_hourly_consumption * 10
    supply = data.expected_pumping_duration * data.expected_pumping_rate
    buffer = data.tank_level_8pm + supply - demand - data.leakage_estimate * 10
    return pd.DataFrame([{
        "tank_level_8pm": data.tank_level_8pm,
        "connected_households": data.connected_households,
        "avg_hourly_consumption": data.avg_hourly_consumption,
        "expected_pumping_duration": data.expected_pumping_duration,
        "expected_pumping_rate": data.expected_pumping_rate,
        "leakage_estimate": data.leakage_estimate,
        "day_type": data.day_type,
        "expected_overnight_demand": demand,
        "expected_pump_supply": supply,
        "projected_water_buffer": buffer
    }])

@app.get("/")
def root():
    return {"message": "Water shortage prediction service is running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_exists": MODEL_PATH.exists()}

@app.post("/predict")
def predict(data: EveningConditions):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model not trained. Run python src/pipeline.py first.")
    try:
        model = joblib.load(MODEL_PATH)
        features = make_features(data)
        prediction = int(model.predict(features)[0])
        label = "Shortage Risk" if prediction == 1 else "Adequate Reserve"
        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(features)[0][1])
        return {
            "prediction": prediction,
            "status": label,
            "shortage_probability": probability,
            "threshold_litres": 15000,
            "prediction_window": "6:00 AM"
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
