"""
test_api.py — API request/response shape tests using FastAPI TestClient.

Tests:
  1. GET /health returns 200 with {"status": "ok", "version": ...}
  2. POST /prediction returns the correct response shape with expected edge_ids.
  3. POST /prediction rejects malformed input (missing required fields).
  4. POST /anomaly returns per-edge anomaly results with required fields.
  5. POST /prediction with multiple edges returns one prediction per edge.
"""

import pytest
from fastapi.testclient import TestClient

# We need a trained model on disk before the API can serve predictions.
# This fixture trains it if not already present.
@pytest.fixture(scope="module", autouse=True)
def ensure_model_trained():
    """Ensure the baseline model is trained before API tests run."""
    from prediction.config import BASELINE_MODEL_PATH
    if not BASELINE_MODEL_PATH.exists():
        from prediction.evaluator import run_evaluation
        run_evaluation(save_artifacts=True)


@pytest.fixture(scope="module")
def client():
    from prediction.api import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_shape(self, client):
        resp = client.get("/health")
        body = resp.json()
        assert "status" in body
        assert body["status"] == "ok"
        assert "version" in body


# ---------------------------------------------------------------------------
# POST /prediction
# ---------------------------------------------------------------------------

SAMPLE_EDGE = {
    "edge_id": "E14",
    "speed": 31.2,
    "flow": 742.0,
    "timestamp": 600,
    "occupancy": 0.3,
}

SAMPLE_REQUEST = {
    "edges": [SAMPLE_EDGE],
    "horizons": [5, 10, 15],
}


class TestPrediction:

    def test_prediction_returns_200(self, client):
        resp = client.post("/prediction", json=SAMPLE_REQUEST)
        assert resp.status_code == 200, f"Body: {resp.text}"

    def test_prediction_response_has_predictions_key(self, client):
        resp = client.post("/prediction", json=SAMPLE_REQUEST)
        body = resp.json()
        assert "predictions" in body, f"Missing 'predictions' key in response: {body}"

    def test_prediction_one_result_per_edge(self, client):
        request = {
            "edges": [
                {"edge_id": "E01", "speed": 25.0, "flow": 400.0, "timestamp": 600},
                {"edge_id": "E02", "speed": 40.0, "flow": 200.0, "timestamp": 600},
                {"edge_id": "E03", "speed": 10.0, "flow": 800.0, "timestamp": 600},
            ],
            "horizons": [5, 10, 15],
        }
        resp = client.post("/prediction", json=request)
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["predictions"]) == 3

    def test_prediction_result_contains_required_fields(self, client):
        resp = client.post("/prediction", json=SAMPLE_REQUEST)
        pred = resp.json()["predictions"][0]
        for field in ["edge_id", "t5", "t10", "t15", "congestion"]:
            assert field in pred, f"Missing field '{field}' in prediction result"

    def test_prediction_edge_id_preserved(self, client):
        resp = client.post("/prediction", json=SAMPLE_REQUEST)
        pred = resp.json()["predictions"][0]
        assert pred["edge_id"] == "E14"

    def test_prediction_congestion_in_range(self, client):
        resp = client.post("/prediction", json=SAMPLE_REQUEST)
        cong = resp.json()["predictions"][0]["congestion"]
        assert 0.0 <= cong <= 1.0, f"Congestion out of [0,1]: {cong}"

    def test_prediction_rejects_missing_required_field(self, client):
        bad_request = {
            "edges": [{"edge_id": "E01", "flow": 400.0, "timestamp": 600}],  # missing speed
            "horizons": [5, 10, 15],
        }
        resp = client.post("/prediction", json=bad_request)
        assert resp.status_code == 422, (
            f"Expected 422 Unprocessable Entity for missing 'speed', got {resp.status_code}"
        )

    def test_prediction_rejects_empty_edges(self, client):
        bad_request = {"edges": [], "horizons": [5, 10, 15]}
        resp = client.post("/prediction", json=bad_request)
        assert resp.status_code == 422

    def test_prediction_numeric_values_are_floats(self, client):
        resp = client.post("/prediction", json=SAMPLE_REQUEST)
        pred = resp.json()["predictions"][0]
        for field in ["t5", "t10", "t15", "congestion"]:
            assert isinstance(pred[field], (int, float)), (
                f"Expected numeric for '{field}', got {type(pred[field])}"
            )


# ---------------------------------------------------------------------------
# POST /anomaly
# ---------------------------------------------------------------------------

ANOMALY_REQUEST = {
    "edges": [
        {"edge_id": "E01", "speed": 5.0, "flow": 900.0, "timestamp": 600},
        {"edge_id": "E02", "speed": 48.0, "flow": 100.0, "timestamp": 600},
    ],
}


class TestAnomaly:

    def test_anomaly_returns_200(self, client):
        resp = client.post("/anomaly", json=ANOMALY_REQUEST)
        assert resp.status_code == 200

    def test_anomaly_response_has_anomalies_key(self, client):
        resp = client.post("/anomaly", json=ANOMALY_REQUEST)
        body = resp.json()
        assert "anomalies" in body

    def test_anomaly_one_result_per_edge(self, client):
        resp = client.post("/anomaly", json=ANOMALY_REQUEST)
        body = resp.json()
        assert len(body["anomalies"]) == len(ANOMALY_REQUEST["edges"])

    def test_anomaly_result_contains_required_fields(self, client):
        resp = client.post("/anomaly", json=ANOMALY_REQUEST)
        result = resp.json()["anomalies"][0]
        for field in ["edge_id", "is_anomaly", "severity"]:
            assert field in result, f"Missing field '{field}' in anomaly result"

    def test_anomaly_is_anomaly_is_bool(self, client):
        resp = client.post("/anomaly", json=ANOMALY_REQUEST)
        for item in resp.json()["anomalies"]:
            assert isinstance(item["is_anomaly"], bool)

    def test_anomaly_severity_in_range(self, client):
        resp = client.post("/anomaly", json=ANOMALY_REQUEST)
        for item in resp.json()["anomalies"]:
            assert 0.0 <= item["severity"] <= 1.0, (
                f"Severity out of [0,1]: {item['severity']}"
            )
