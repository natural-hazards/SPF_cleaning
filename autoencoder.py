# autoencoder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


def _as_2d_float64(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {X.shape}")
    if not np.issubdtype(X.dtype, np.floating):
        X = X.astype(np.float64)
    return X


class MLPAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden1: int, hidden2: int, latent_dim: int):
        super().__init__()

        def ln(d: int) -> nn.LayerNorm:
            return nn.LayerNorm(d)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            ln(hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden1, hidden2),
            ln(hidden2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden2),
            ln(hidden2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden2, hidden1),
            ln(hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden1, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


@dataclass
class AEConfig:
    # model shape
    latent_dim: int = 32

    # training
    epochs: int = 60
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5

    # robust loss
    huber_delta: float = 1.0

    # denoising (recommended)
    noise_std: float = 0.01         # additive Gaussian noise on inputs during training
    feature_dropout: float = 0.02   # randomly zero a fraction of features during training

    # thresholding
    val_split: float = 0.2
    threshold_percentile: float = 99.0  # percentile of OK validation reconstruction errors

    # reproducibility
    seed: int = 0

    # device
    device: str = "cpu"


class AutoencoderCleaner:
    """
    Train an autoencoder on OK feature vectors (KMeans-format FFT features).
    Use reconstruction error as anomaly score.

    Typical usage:
        cleaner = AutoencoderCleaner()
        cleaner.fit(X_ok)
        score = cleaner.score(X_new)
        is_ok = cleaner.is_ok(x_one)
    """

    def __init__(self, config: Optional[AEConfig] = None):
        self.cfg = config or AEConfig()
        self.model: Optional[MLPAutoencoder] = None
        self.threshold_: Optional[float] = None
        self.input_dim_: Optional[int] = None

        # set seeds
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

    @staticmethod
    def _choose_widths(input_dim: int) -> Tuple[int, int]:
        # conservative defaults for FFT features
        if input_dim <= 3000:
            return 512, 128
        else:
            return 1024, 256

    def _make_dataloader(self, X: np.ndarray, shuffle: bool) -> torch.utils.data.DataLoader:
        tensor = torch.from_numpy(X.astype(np.float32, copy=False))
        ds = torch.utils.data.TensorDataset(tensor)
        return torch.utils.data.DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=shuffle, drop_last=False)

    def fit(self, X_ok: np.ndarray) -> "AutoencoderCleaner":
        X_ok = _as_2d_float64(X_ok)

        if np.isnan(X_ok).any():
            raise ValueError("X_ok contains NaNs. Filter these before training the autoencoder.")

        n, d = X_ok.shape
        if n < 10:
            raise ValueError(f"Need more OK samples to train AE reliably; got {n}.")

        self.input_dim_ = d
        h1, h2 = self._choose_widths(d)
        self.model = MLPAutoencoder(input_dim=d, hidden1=h1, hidden2=h2, latent_dim=self.cfg.latent_dim).to(self.cfg.device)

        # split OK into train/val
        idx = np.arange(n)
        np.random.shuffle(idx)
        n_val = max(1, int(round(self.cfg.val_split * n)))
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:] if n_val < n else idx[:0]

        if tr_idx.size < 5:
            # if tiny, just use all for training but threshold on same (not ideal, but works)
            tr_idx = idx
            val_idx = idx

        X_tr = X_ok[tr_idx]
        X_val = X_ok[val_idx]

        tr_loader = self._make_dataloader(X_tr, shuffle=True)
        val_loader = self._make_dataloader(X_val, shuffle=False)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        # Huber is robust; for very clean data MSE also works
        loss_fn = nn.HuberLoss(delta=self.cfg.huber_delta, reduction="mean")

        self.model.train()
        for _epoch in range(self.cfg.epochs):
            for (xb,) in tr_loader:
                xb = xb.to(self.cfg.device)

                # denoising: add small noise + feature dropout (only during training)
                x_in = xb
                if self.cfg.noise_std > 0:
                    x_in = x_in + self.cfg.noise_std * torch.randn_like(x_in)
                if self.cfg.feature_dropout > 0:
                    mask = (torch.rand_like(x_in) > self.cfg.feature_dropout).float()
                    x_in = x_in * mask

                optimizer.zero_grad(set_to_none=True)
                x_hat = self.model(x_in)
                loss = loss_fn(x_hat, xb)  # target is clean xb
                loss.backward()
                optimizer.step()

        # threshold from validation OK reconstruction errors
        val_scores = self._score_loader(val_loader)
        self.threshold_ = float(np.percentile(val_scores, self.cfg.threshold_percentile))
        return self

    @torch.no_grad()
    def _score_loader(self, loader: torch.utils.data.DataLoader) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not trained. Call fit() first.")

        self.model.eval()
        scores = []
        for (xb,) in loader:
            xb = xb.to(self.cfg.device)
            x_hat = self.model(xb)
            # MAE per sample tends to be robust and easy to interpret
            err = torch.mean(torch.abs(x_hat - xb), dim=1)
            scores.append(err.detach().cpu().numpy())
        return np.concatenate(scores, axis=0)

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Returns reconstruction error (anomaly score). Higher => more anomalous.
        """
        if self.model is None or self.threshold_ is None or self.input_dim_ is None:
            raise RuntimeError("Cleaner not fitted. Call fit(X_ok) first.")
        X = _as_2d_float64(X)
        if X.shape[1] != self.input_dim_:
            raise ValueError(f"Input dim mismatch: expected {self.input_dim_}, got {X.shape[1]}")
        if np.isnan(X).any():
            # return inf for NaN rows rather than crashing
            scores = np.full((X.shape[0],), np.inf, dtype=np.float64)
            ok_rows = ~np.isnan(X).any(axis=1)
            if np.any(ok_rows):
                scores[ok_rows] = self._score_numpy(X[ok_rows])
            return scores
        return self._score_numpy(X)

    def _score_numpy(self, X: np.ndarray) -> np.ndarray:
        loader = self._make_dataloader(X, shuffle=False)
        return self._score_loader(loader).astype(np.float64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns boolean array: True => corrupted/anomalous.
        """
        s = self.score(X)
        return s > float(self.threshold_)

    def is_ok(self, x: np.ndarray) -> bool:
        """
        Convenience for single vector.
        """
        s = float(self.score(x)[0])
        return s <= float(self.threshold_)

    def state_dict(self) -> dict:
        """
        Serialize model + threshold + config.
        """
        if self.model is None:
            raise RuntimeError("No model to save (fit first).")
        return {
            "cfg": self.cfg.__dict__,
            "input_dim": self.input_dim_,
            "threshold": self.threshold_,
            "model_state": self.model.state_dict(),
        }

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, map_location: Optional[str] = "cpu") -> "AutoencoderCleaner":
        blob = torch.load(path, map_location=map_location)
        cfg = AEConfig(**blob["cfg"])
        cfg.device = map_location or cfg.device
        cleaner = cls(cfg)
        cleaner.input_dim_ = int(blob["input_dim"])
        cleaner.threshold_ = float(blob["threshold"])
        h1, h2 = cleaner._choose_widths(cleaner.input_dim_)
        cleaner.model = MLPAutoencoder(
            input_dim=cleaner.input_dim_,
            hidden1=h1,
            hidden2=h2,
            latent_dim=cleaner.cfg.latent_dim,
        ).to(cleaner.cfg.device)
        cleaner.model.load_state_dict(blob["model_state"])
        cleaner.model.eval()
        return cleaner
