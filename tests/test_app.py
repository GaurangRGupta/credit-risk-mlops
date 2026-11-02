import pandas as pd
import pytest

from src.app import app

SAMPLE = pd.read_csv("feature_store/credit_features.csv").drop(
    columns=["risk", "credit_per_month"]).iloc[0].to_dict()


@pytest.fixture
def client():
    return app.test_client()


def test_health(client):
    assert client.get("/health").get_json() == {"status": "ok"}


def test_predict_valid(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.get_json()
    assert body["risk"] in {"good", "bad"}
    assert 0 <= body["probability_bad"] <= 1


def test_predict_missing_fields(client):
    r = client.post("/predict", json={"duration": 12})
    assert r.status_code == 400


def test_metrics(client):
    client.post("/predict", json=SAMPLE)
    assert b"predictions_total" in client.get("/metrics").data


def test_ui_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Credit Risk Predictor" in r.data
