import pytest
from fastapi.testclient import TestClient
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_predict():
    response = client.post("/predict", json={"features": [1, 2, 3, 4]})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "drift_detected" in data

def test_drift_detection():
    response = client.post("/predict", json={"features": [100, 200, 300, 400]})
    assert response.status_code == 200
    data = response.json()
    assert data["drift_detected"] == True

def test_stats():
    response = client.get("/stats")
    assert response.status_code == 200
