import joblib
from fastapi.testclient import TestClient

from house_price_lab import api


class ConstantModel:
    def predict(self, rows):
        return [123_456.789]


def test_health():
    response = TestClient(api.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction_returns_model_output(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"
    joblib.dump(ConstantModel(), model_path)
    monkeypatch.setattr(api, "MODEL_PATH", model_path)

    response = TestClient(api.app).post(
        "/predict",
        json={
            "area_sqft": 900,
            "bedrooms": 2,
            "age_years": 10,
            "location": "suburban",
            "property_type": "house",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"predicted_price": 123456.79}


def test_prediction_rejects_unknown_categories():
    response = TestClient(api.app).post(
        "/predict",
        json={
            "area_sqft": 900,
            "bedrooms": 2,
            "age_years": 10,
            "location": "moon",
            "property_type": "house",
        },
    )
    assert response.status_code == 422


def test_prediction_without_model_is_a_clear_503(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "missing.joblib")
    response = TestClient(api.app).post(
        "/predict",
        json={
            "area_sqft": 900,
            "bedrooms": 2,
            "age_years": 10,
            "location": "suburban",
            "property_type": "house",
        },
    )
    assert response.status_code == 503
