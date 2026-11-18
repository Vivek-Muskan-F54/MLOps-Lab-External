import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from app.main import app
from src.pipeline import generate_dataset, engineer_features, CAPACITY

client = TestClient(app)

def test_dataset_has_250_rows_and_valid_tank_levels():
    df = generate_dataset(n=250)
    assert len(df) == 250
    assert df["tank_level_8pm"].between(0, CAPACITY).all()

def test_tank_level_cannot_exceed_capacity():
    df = generate_dataset()
    assert (df["tank_level_8pm"] <= CAPACITY).all()
    with pytest.raises(Exception):
        client.post("/predict", json={
            "tank_level_8pm": 100001,
            "connected_households": 200,
            "avg_hourly_consumption": 10,
            "expected_pumping_duration": 1,
            "expected_pumping_rate": 10000,
            "leakage_estimate": 50,
            "day_type": "Weekday"
        })

def test_feature_engineering():
    df = pd.DataFrame([{
        "tank_level_8pm": 50000, "connected_households": 100,
        "avg_hourly_consumption": 10, "expected_pumping_duration": 1,
        "expected_pumping_rate": 10000, "leakage_estimate": 100,
        "day_type": "Weekday", "morning_water_shortage": 0
    }])
    result = engineer_features(df)
    assert result.loc[0, "expected_overnight_demand"] == 10000
    assert result.loc[0, "expected_pump_supply"] == 10000
    assert result.loc[0, "projected_water_buffer"] == 49000

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
