"""Tests de l'API FastAPI Heart Disease"""
import pytest
import httpx

@pytest.fixture
def api():
    return "http://localhost:8000"

def test_health(api):
    r = httpx.get(api + "/health")
    assert r.status_code == 200

def test_predict(api):
    data = {
        "features": {
            "age": 50, "sex": 1, "cp": 2, "trestbps": 120, "chol": 250,
            "fbs": 0, "restecg": 0, "thalach": 160, "exang": 0,
            "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 2
        }
    }
    r = httpx.post(api + "/predict", json=data)
    assert r.status_code == 200

def test_predict_missing_features_returns_422(api):
    """Features vides : doit retourner 422"""
    data = {"features": {}}
    r = httpx.post(api + "/predict", json=data)
    assert r.status_code == 422

def test_predict_wrong_type_returns_422(api):
    """Mauvais type (texte au lieu de nombre) : doit retourner 422"""
    data = {
        "features": {
            "age": "pas_un_nombre", "sex": 1, "cp": 2, "trestbps": 120,
            "chol": 250, "fbs": 0, "restecg": 0, "thalach": 160,
            "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 2
        }
    }
    r = httpx.post(api + "/predict", json=data)
    assert r.status_code == 422