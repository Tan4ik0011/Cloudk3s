import os
import pytest

# Настройки окружения должны быть заданы ДО импорта FastAPI-приложения
os.environ["BYPASS_KEYCLOAK"] = "true"
os.environ["MODEL_PATH"] = "models/cvae_model.pt"
os.environ["SCALER_PATH"] = "models/scaler.pkl"

from fastapi.testclient import TestClient
from src.main import app

@pytest.fixture
def client():
    # Использование контекстного менеджера гарантирует вызов событий startup
    with TestClient(app) as c:
        yield c