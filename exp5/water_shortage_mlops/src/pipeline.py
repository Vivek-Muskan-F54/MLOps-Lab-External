from pathlib import Path
import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FEATURES = ROOT / "feature_store"
ARTIFACTS = ROOT / "artifacts"
for p in (DATA, FEATURES, ARTIFACTS):
    p.mkdir(parents=True, exist_ok=True)

CAPACITY = 100_000
HOURS = 10  # 8:00 PM to 6:00 AM

def generate_dataset(seed=42, summer=False, n=250):
    rng = np.random.default_rng(seed + (100 if summer else 0))
    rows = []
    for i in range(n):
        households = int(rng.integers(100, 401))
        base_consumption = rng.uniform(7.0, 15.0)
        consumption = base_consumption * (1.35 if summer else 1.0)
        # Small night-to-night variation around the seasonal mean.
        consumption *= rng.uniform(0.88, 1.12)
        level = float(rng.uniform(25_000, 95_000))
        duration = float(rng.choice([0, 0.5, 1, 1.5, 2, 2.5, 3]))
        pump_rate = float(rng.uniform(8_000, 18_000))
        leakage = float(rng.uniform(0, 250))
        day_type = str(rng.choice(["Weekday", "Weekend"], p=[5/7, 2/7]))

        demand = households * consumption * HOURS
        supply = duration * pump_rate
        leak_volume = leakage * HOURS
        # Add modest unobserved variation to represent uncertain actual demand/supply.
        actual_demand = demand * rng.uniform(0.92, 1.12)
        actual_supply = supply * rng.uniform(0.90, 1.05)
        actual_leak = leak_volume * rng.uniform(0.85, 1.20)
        morning_level = np.clip(level + actual_supply - actual_demand - actual_leak, 0, CAPACITY)
        target = int(morning_level < 0.15 * CAPACITY)

        rows.append({
            "night_id": f"{'SUM' if summer else 'NOR'}-{i+1:03d}",
            "tank_level_8pm": round(level, 2),
            "connected_households": households,
            "avg_hourly_consumption": round(consumption, 2),
            "expected_pumping_duration": duration,
            "expected_pumping_rate": round(pump_rate, 2),
            "leakage_estimate": round(leakage, 2),
            "day_type": day_type,
            "morning_water_shortage": target
        })
    return pd.DataFrame(rows)

def create_datasets():
    v1 = generate_dataset(seed=42, summer=False)
    v2 = generate_dataset(seed=42, summer=True)
    v1.to_csv(DATA / "water_data_v1.csv", index=False)
    v2.to_csv(DATA / "water_data_v2_summer.csv", index=False)
    manifest = {
        "v1": "250 synthetic nights; normal household demand",
        "v2": "250 synthetic nights; household consumption increased by 35% before nightly variation",
        "capacity_litres": CAPACITY,
        "shortage_threshold_litres": 15000,
        "prediction_window": "8:00 PM to 6:00 AM (10 hours)"
    }
    (DATA / "version_manifest.json").write_text(json.dumps(manifest, indent=2))
    return v1, v2

def engineer_features(df):
    out = df.copy()
    out["expected_overnight_demand"] = (
        out["connected_households"] * out["avg_hourly_consumption"] * HOURS
    )
    out["expected_pump_supply"] = (
        out["expected_pumping_duration"] * out["expected_pumping_rate"]
    )
    out["projected_water_buffer"] = (
        out["tank_level_8pm"] + out["expected_pump_supply"]
        - out["expected_overnight_demand"] - out["leakage_estimate"] * HOURS
    )
    return out

def train_and_register():
    v1, v2 = create_datasets()
    combined = pd.concat([v1, v2], ignore_index=True)
    featured = engineer_features(combined)
    # Persist reusable feature data separately; label and night_id retained for traceability.
    featured.to_csv(FEATURES / "engineered_features.csv", index=False)

    target = "morning_water_shortage"
    numeric = [
        "tank_level_8pm", "connected_households", "avg_hourly_consumption",
        "expected_pumping_duration", "expected_pumping_rate", "leakage_estimate",
        "expected_overnight_demand", "expected_pump_supply", "projected_water_buffer"
    ]
    categorical = ["day_type"]
    X = featured[numeric + categorical]
    y = featured[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])
    models = {
        "DecisionTree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42)
    }

    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("overnight-water-shortage")
    results = []
    best = None
    # Prioritize recall: missing a real shortage is more costly than a false alarm.
    for name, classifier in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        precision = precision_score(y_test, pred, zero_division=0)
        recall = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        cm = confusion_matrix(y_test, pred, labels=[0, 1]).tolist()
        results.append({
            "model": name, "precision": precision, "recall": recall,
            "f1_score": f1, "confusion_matrix": cm
        })
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params({"model": name, "test_size": 0.25, "random_state": 42})
            mlflow.log_metrics({"precision": precision, "recall": recall, "f1_score": f1})
            mlflow.log_dict({"confusion_matrix_labels": ["Adequate", "Shortage"], "matrix": cm},
                            "confusion_matrix.json")
            mlflow.sklearn.log_model(pipe, artifact_path="model")
            results[-1]["run_id"] = run.info.run_id
        score = (recall, f1, precision)
        if best is None or score > best["score"]:
            best = {"score": score, "name": name, "pipeline": pipe, "metrics": results[-1]}

    # Register only the selected model.
    with mlflow.start_run(run_name=f"best-{best['name']}") as run:
        mlflow.log_params({"selected_model": best["name"], "selection_priority": "recall_then_f1_then_precision"})
        mlflow.log_metrics({
            "precision": best["metrics"]["precision"],
            "recall": best["metrics"]["recall"],
            "f1_score": best["metrics"]["f1_score"]
        })
        model_info = mlflow.sklearn.log_model(
            best["pipeline"], artifact_path="selected_model",
            registered_model_name="OvernightWaterShortageModel"
        )
    joblib.dump(best["pipeline"], ARTIFACTS / "water_shortage_model.joblib")
    (ARTIFACTS / "model_comparison.json").write_text(json.dumps(results, indent=2))
    (ARTIFACTS / "selected_model.json").write_text(json.dumps({
        "model": best["name"], "registered_model": "OvernightWaterShortageModel",
        "registered_model_uri": model_info.model_uri,
        "selection_rule": "Highest shortage recall; ties broken by F1 then precision",
        "metrics": best["metrics"]
    }, indent=2))
    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nSelected model: {best['name']}")
    print("Saved model: artifacts/water_shortage_model.joblib")
    print("Registered model: OvernightWaterShortageModel")
    return results

if __name__ == "__main__":
    train_and_register()
