import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    mu = x.mean()
    sd = x.std()
    if sd < eps:
        return np.zeros_like(x, dtype=np.float64)
    return (x - mu) / sd

def fix_length(x, target_len):
    """
    x: (T,) 1D signal
    returns (target_len,)
    """
    T = len(x)

    if T == target_len:
        return x

    if T > target_len:
        return x[:target_len]     # crop

    # pad if shorter
    out = np.zeros(target_len, dtype=x.dtype)
    out[:T] = x
    return out


def sliding_windows_fast(x, win_len, hop):
    w = sliding_window_view(x, (win_len, 3))[:, 0]
    return w[::hop]

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


def video_features_from_mseed(mseed, working_length: int, win_len: int, hop: int = 1) -> np.ndarray:
    channels = []
    for i in range(3):
        sig = mseed[i].data.astype(np.float32, copy=False)      # (T,)
        sig = fix_length(sig, working_length)                   # (working_length,)
        sig = zscore(sig).astype(np.float32, copy=False)        # normalize after pad/crop
        channels.append(sig)

    x = np.stack(channels, axis=1)  # (T, 3)
    return sliding_windows_fast(x, win_len=win_len, hop=hop)  # (N, win_len, 3)