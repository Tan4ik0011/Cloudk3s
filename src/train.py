import os
import torch
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from src.model import ConditionalVAE
from src.preprocess import make_windows


def train_and_export():
    print("Старт предобработки данных...")
    raw_path = 'data/daily_dataset.csv'
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Файл daily_dataset.csv отсутствует по пути: {raw_path}")

    # Чтение и очистка данных для MAC000002
    df_raw = pd.read_csv(raw_path)
    df = df_raw[df_raw['LCLid'] == 'MAC000002'].copy()

    df['day'] = pd.to_datetime(df['day'])
    df = df.sort_values('day').reset_index(drop=True)
    df = df.replace('Null', np.nan).dropna(subset=['energy_sum'])

    df['energy_sum'] = pd.to_numeric(df['energy_sum'])
    df['energy_max'] = pd.to_numeric(df['energy_max'])
    df['energy_min'] = pd.to_numeric(df['energy_min'])

    FEATURES = ['energy_max', 'energy_min', 'energy_sum']
    df = df[FEATURES]

    # Нормализация данных
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.values)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    print("Скейлер сохранен в models/scaler.pkl")

    # Превращаем в скользящие окна размером 32
    windows = make_windows(scaled_data, window_size=32)

    # Инициализация архитектуры
    print("Запуск быстрого цикла обучения CVAE на CPU (10 эпох)...")
    model = ConditionalVAE(window_size=32, num_features=3, target_idx=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(windows))
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(1, 11):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon, target, mu, logvar = model(batch)

            recon_loss = ((recon - target).pow(2).sum(dim=1)).mean()
            kl_loss = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=1).mean()
            loss = recon_loss + 0.02 * kl_loss

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        print(f"Эпоха {epoch}/10 - Средние потери: {epoch_loss / len(windows):.4f}")

    torch.save(model.state_dict(), "models/cvae_model.pt")
    print("Веса модели сохранены в models/cvae_model.pt")


if __name__ == "__main__":
    train_and_export()