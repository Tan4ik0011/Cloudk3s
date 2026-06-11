import pytest


def test_healthcheck(client):
    response = client.get("/healthcheck")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_unauthorized_prediction_no_header(client):
    # Отправляем запрос без заголовка Authorization
    # Используем новую структуру данных
    payload = {
        "data": [
            {"energy_max": 1.2, "energy_min": 0.4, "energy_sum": 0.8}
            for _ in range(32)
        ]
    }
    response = client.post("/predictions", json=payload)
    assert response.status_code in [401, 403]


def test_prediction_with_mock_authorized_token(client):
    # ИСПРАВЛЕНО: Генерируем тестовый payload под новый формат JSON-данных
    valid_payload = {
        "data": [
            {"energy_max": 1.2, "energy_min": 0.4, "energy_sum": 0.8}
            for _ in range(32)
        ]
    }

    # Заголовок авторизации (обычно мокается через conftest.py)
    headers = {"Authorization": "Bearer 00000"}
    response = client.post("/predictions", json=valid_payload, headers=headers)

    assert response.status_code == 200
    assert "reconstructed_energy_sum" in response.json()