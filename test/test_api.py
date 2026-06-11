import pytest


def test_healthcheck(client):
    response = client.get("/healthcheck")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_unauthorized_prediction_no_header(client):
    dummy_payload = {"features": [[0.5, 0.1, 0.2] for _ in range(32)]}
    response = client.post("/predictions", json=dummy_payload)
    assert response.status_code in [401, 403]


def test_prediction_with_mock_authorized_token(client):
    valid_payload = {
        "features": [[1.2, 0.4, 0.8] for _ in range(32)]
    }
    headers = {"Authorization": "Bearer 00000"}
    response = client.post("/predictions", json=valid_payload, headers=headers)
    assert response.status_code == 200

    data = response.json()
    assert "reconstructed_energy_sum" in data
    assert len(data["reconstructed_energy_sum"]) == 32