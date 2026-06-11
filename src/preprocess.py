import numpy as np

def make_windows(series: np.ndarray, window_size: int) -> np.ndarray:
    windows = np.lib.stride_tricks.sliding_window_view(series, window_shape=window_size, axis=0)
    return np.moveaxis(windows, -1, 1).astype(np.float32)

def inverse_scale_feature(x: np.ndarray, scaler, feature_pos: int) -> np.ndarray:
    x = np.asarray(x)
    full = np.zeros((x.reshape(-1).shape[0], scaler.n_features_in_), dtype=np.float32)
    full[:, feature_pos] = x.reshape(-1)
    restored = scaler.inverse_transform(full)
    return restored[:, feature_pos].reshape(x.shape)