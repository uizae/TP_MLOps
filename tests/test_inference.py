"""Test inférence"""
from mlops_tp.inference import predict

def test_predict_returns_valid_class():
    features = {
        "age": 50, "sex": 1, "cp": 2, "trestbps": 120, "chol": 250,
        "fbs": 0, "restecg": 0, "thalach": 160, "exang": 0,
        "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 2
    }
    result = predict(features)
    assert result["prediction"] in [0, 1]

def test_predict_proba_between_0_and_1():
    features = {
        "age": 50, "sex": 1, "cp": 2, "trestbps": 120, "chol": 250,
        "fbs": 0, "restecg": 0, "thalach": 160, "exang": 0,
        "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 2
    }
    result = predict(features)
    assert 0.0 <= result["proba"]["no_disease"] <= 1.0
    assert 0.0 <= result["proba"]["disease"] <= 1.0
    assert abs(result["proba"]["no_disease"] + result["proba"]["disease"] - 1.0) < 1e-6