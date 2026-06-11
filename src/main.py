import os
import torch
import numpy as np
import joblib
import requests
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List

from src.model import ConditionalVAE
from src.preprocess import inverse_scale_feature

app = FastAPI(title="CVAE Inference Service", version="1.0")
security = HTTPBearer()

# Считывание путей к артефактам
MODEL_PATH = os.getenv("MODEL_PATH", "models/cvae_model.pt")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")

# Конфигурация Keycloak (Lab 2)
KEYCLOAK_URL = os.getenv("KEYCLOAK_URL", "http://keycloak:8080")
KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM", "inference")
KEYCLOAK_CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID", "inference-client")
KEYCLOAK_CLIENT_SECRET = os.getenv("KEYCLOAK_CLIENT_SECRET", "")
BYPASS_KEYCLOAK = os.getenv("BYPASS_KEYCLOAK", "true").lower() == "true"

model = None
scaler = None


@app.on_event("startup")
def load_artifacts():
    global model, scaler
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise RuntimeError(
            f"Файлы моделей не найдены по путям {MODEL_PATH} или {SCALER_PATH}. "
            f"Запустите локальное обучение: 'python src/train.py'."
        )
    scaler = joblib.load(SCALER_PATH)
    model = ConditionalVAE(window_size=32, num_features=3, target_idx=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()


# Новая модель для валидации отдельного временного шага
class FeatureItem(BaseModel):
    energy_max: float = Field(..., description="Максимальное потребление энергии за день")
    energy_min: float = Field(..., description="Минимальное потребление энергии за день")
    energy_sum: float = Field(..., description="Суммарное потребление энергии за день")


# Обновленная модель входящего окна
class WindowInput(BaseModel):
    data: List[FeatureItem] = Field(..., description="Массив из 32 структурированных объектов")


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials

    # Сценарий локальной проверки / обхода (Lab 1)
    if BYPASS_KEYCLOAK and token == "00000":
        return {"active": True, "scope": "doInfer"}

    # Сценарий интеграции с Keycloak (Lab 2)
    introspection_url = f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/token/introspect"
    data = {
        "token": token,
        "client_id": KEYCLOAK_CLIENT_ID,
        "client_secret": KEYCLOAK_CLIENT_SECRET
    }
    try:
        response = requests.post(introspection_url, data=data, timeout=5)
        if response.status_code != 200:
            raise HTTPException(status_code=401, detail="Интроспекция токена не удалась")

        res_json = response.json()
        if not res_json.get("active"):
            raise HTTPException(status_code=401, detail="Токен неактивен или недействителен")

        # Проверка прав доступа Keycloak (scope: doInfer)
        scope = res_json.get("scope", "")
        if "doInfer" not in scope:
            raise HTTPException(status_code=403, detail="Доступ запрещен: отсутствует область видимости 'doInfer'")

        return res_json
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Сервис аутентификации недоступен: {str(e)}")


@app.get("/healthcheck")
def healthcheck():
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Модели не загружены")
    return {"status": "healthy"}


@app.post("/predictions")
def predict(payload: WindowInput, auth_info: dict = Depends(verify_token)):
    if len(payload.data) != 32:
        raise HTTPException(status_code=400, detail="Входной массив 'data' должен содержать ровно 32 элемента")

    # Формируем двумерный массив строго в порядке признаков FEATURES = ['energy_max', 'energy_min', 'energy_sum']
    features_list = []
    for item in payload.data:
        features_list.append([item.energy_max, item.energy_min, item.energy_sum])

    input_array = np.array(features_list, dtype=np.float32)

    try:
        scaled_data = scaler.transform(input_array)
        tensor_data = torch.from_numpy(scaled_data).unsqueeze(0)

        with torch.no_grad():
            recon, _, _, _ = model(tensor_data, sample=False)
            recon_np = recon.numpy()[0]

        reconstructed_target = inverse_scale_feature(recon_np, scaler, feature_pos=2)

        return {
            "reconstructed_energy_sum": reconstructed_target.tolist(),
            "scope": auth_info.get("scope", "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")