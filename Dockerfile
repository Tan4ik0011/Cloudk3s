FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

EXPOSE 8000

ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/cvae_model.pt
ENV SCALER_PATH=/app/models/scaler.pkl

ENV KEYCLOAK_URL=http://keycloak:8080
ENV KEYCLOAK_REALM=inference
ENV KEYCLOAK_CLIENT_ID=inference-client
ENV KEYCLOAK_CLIENT_SECRET=change-me
ENV BYPASS_KEYCLOAK=true

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]