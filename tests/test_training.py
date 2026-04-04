"""Test modèle existant"""
from pathlib import Path
import joblib
import json

MODEL_PATH = "src/mlops_tp/artifacts/model.joblib"
METRICS_PATH = "src/mlops_tp/artifacts/metrics.json"

def test_model():
    assert Path(MODEL_PATH).exists()

def test_predict():
    model = joblib.load(MODEL_PATH)
    import pandas as pd
    data = pd.DataFrame([{
        "age": 50, "sex": 1, "cp": 2, "trestbps": 120, "chol": 250,
        "fbs": 0, "restecg": 0, "thalach": 160, "exang": 0, 
        "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 2
    }])
    pred = model.predict(data)[0]
    assert pred in [0,1]
    
def test_metrics():
    with open(METRICS_PATH) as f:
        m = json.load(f)
    assert m["accuracy_val"] > 0.6