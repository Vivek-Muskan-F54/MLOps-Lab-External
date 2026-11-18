# Overnight Water Shortage Prediction – MLOps

Predict whether a residential water tank will fall below 15% (15,000 L) by 6 AM using conditions available at 8 PM.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Train, compare and register
```bash
python src/pipeline.py
```
This creates two 250-row synthetic dataset versions, a reusable feature-store CSV, evaluation JSON files, a local joblib model, and MLflow tracking/registry data.

## Start MLflow UI
```bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```
Open http://127.0.0.1:5000

## Start FastAPI
```bash
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000/docs or call POST /predict.

## Example prediction request
```json
{
  "tank_level_8pm": 65000,
  "connected_households": 200,
  "avg_hourly_consumption": 10,
  "expected_pumping_duration": 1,
  "expected_pumping_rate": 12000,
  "leakage_estimate": 50,
  "day_type": "Weekday"
}
```

## Tests
```bash
pytest -v
```
GitHub Actions runs these tests on pushes and pull requests to main/master.

## Dataset versions
- `data/water_data_v1.csv`: normal-demand scenario.
- `data/water_data_v2_summer.csv`: summer scenario, base household demand multiplied by 1.35.
- `data/version_manifest.json`: records the version descriptions and assumptions.

## Feature store
`feature_store/engineered_features.csv` contains original input columns, engineered features and target for reuse. The model pipeline is trained with preprocessing fit only on the training split.

## Important
Synthetic data is for laboratory demonstration only and does not represent measurements from a real water utility. Model metrics must be taken from the actual run.
