# MLOps Implementation Report
## Predicting Overnight Water Shortages in a Residential Water Tank

**Student Name:** ____________________  
**Roll Number:** ____________________  
**Course:** MLOps Laboratory  
**Date:** ____________________

## 1. Problem Statement
Build a binary classification system that predicts whether the tank level will be below 15,000 litres by 6:00 AM, using observations available at 8:00 PM.

## 2. Objectives
- Generate and version 250-row synthetic datasets.
- Create normal and summer demand scenarios.
- Preprocess data and engineer reusable features.
- Train and compare Decision Tree and Random Forest.
- Track and register the selected model with MLflow.
- Automate tests with GitHub Actions.
- Serve predictions through FastAPI.

## 3. Dataset and Versioning
Tank capacity: 100,000 L. Shortage threshold: 15,000 L. Prediction window: 10 hours.

Version 1 represents normal conditions. Version 2 represents summer conditions with a 1.35 multiplier applied to base household consumption. Both versions contain 250 records. The manifest documents the changes.

## 4. Data Pipeline and Feature Engineering
Numeric missing values are imputed using the median and scaled using StandardScaler. Day type is imputed using the most frequent category and one-hot encoded. These transformations are in a Scikit-learn ColumnTransformer and Pipeline.

Engineered features:
- expected_overnight_demand = households × average hourly consumption × 10
- expected_pump_supply = pumping duration × pumping rate
- projected_water_buffer = evening level + pump supply − overnight demand − estimated leakage × 10

The engineered feature table is saved at `feature_store/engineered_features.csv`.

## 5. Model Training and Evaluation
Two classifiers are trained: Decision Tree and Random Forest. The test set is 25% of the data and stratified. Metrics include precision, recall, F1-score and confusion matrix. The selected model prioritizes shortage recall, followed by F1-score and precision, because missing a shortage is operationally costly.

**Paste actual output from `artifacts/model_comparison.json` here.**

| Model | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Decision Tree | ______ | ______ | ______ |
| Random Forest | ______ | ______ | ______ |

Selected model: ____________________

## 6. MLflow Tracking and Registry
Experiment name: `overnight-water-shortage`  
Registered model: `OvernightWaterShortageModel`

MLflow logs model parameters, precision, recall, F1-score and confusion matrix. Include a screenshot of the MLflow experiment runs and registered model.

## 7. CI/CD and Testing
GitHub Actions workflow: `.github/workflows/ci.yml`. It installs dependencies and executes Pytest on pushes and pull requests. Tests validate dataset row count, tank level limits, feature calculations and API health endpoint.

Include screenshot of a successful GitHub Actions run.

## 8. FastAPI Model Serving
The selected model is served through FastAPI. Endpoint: `POST /predict`. The request contains evening tank conditions. The response includes prediction, status and shortage probability.

Include actual request and response screenshots from `/docs` or an API client.

## 9. Results and Conclusion
The project demonstrates a complete synthetic-data MLOps lifecycle: dataset versioning, preprocessing, feature engineering, model comparison, MLflow registration, automated testing and API serving.

The model is a laboratory prototype trained on synthetic data. Real deployment would require measured tank telemetry, validation across seasons, monitoring and periodic retraining.

## 10. GitHub Repository
Repository URL: ____________________
