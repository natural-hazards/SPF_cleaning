import numpy as np

def zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    mu = x.mean()
    sd = x.std()
    if sd < eps:
        return np.zeros_like(x, dtype=np.float64)
    return (x - mu) / sd

def fft_features(x: np.ndarray, n_fft: int, use_log: bool = True) -> np.ndarray:
    X = np.fft.rfft(x, n=n_fft)  # pad or trim here
    mag = np.abs(X)
    if use_log:
        mag = np.log1p(mag)
    return mag

def event_features_from_mseed(mseed, n_fft: int) -> np.ndarray:
    """
    z-score -> fixed length -> FFT mag -> concat.
    """
    transformed_data = []
    for i in range(3):
        sig = mseed[i].data
        sig = zscore(sig)
        transformed_data.append(fft_features(sig, n_fft=n_fft, use_log=True))
    return np.concatenate(transformed_data, axis=0)