import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_list_models_empty():
    # Because tests run against an empty DB initially
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_list_experiments_empty():
    response = client.get("/api/v1/experiments")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
