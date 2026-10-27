"""nhits.py - the N-HiTS quantile network (PyTorch), its training loop, and a wrapper that plugs it into the backtest."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .windows import WindowBuilder, WindowedForecaster


def make_tensors(data):
    # Responsibility: numpy float64 arrays -> torch float32 tensors (the dtype layers expect).
    # Example: data["X"] shape (6500, 6) -> tensor (6500, 6), float32
    X = torch.tensor(data["X"], dtype=torch.float32)       # (n, L)  log1p windows
    M = torch.tensor(data["M"], dtype=torch.float32)       # (n, L)  1 = real month, 0 = padding
    C = torch.tensor(data["C"], dtype=torch.float32)       # (n, F)  covariates
    return X, M, C


def pinball(pred, y, tau):
    # Responsibility: the quantile loss, averaged over rows and levels.
    # Formula:  tau * d if d >= 0 else (tau - 1) * d,  with d = actual - predicted
    # Example:  predictions [1.0, 3.2, 4.2] at tau [0.1, 0.5, 0.9]: actual 4.5109 -> 0.4288;  actual 0 -> 0.9733
    d = y.unsqueeze(1) - pred                              # (n, n_levels): actual minus predicted quantile
    return torch.maximum(tau * d, (tau - 1) * d).mean()


class Block(nn.Module):
    """One N-HiTS block: pool the window, run an MLP, output a backcast (what it explains) and a forecast (its contribution)."""

    def __init__(self, L, n_cov, n_levels, pool, hidden=64):
        super().__init__()
        assert L % pool == 0, "window length must be divisible by the pool size"
        self.pool = pool
        self.mlp = nn.Sequential(nn.Linear(2 * (L // pool) + n_cov, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.back = nn.Linear(hidden, L)                   # backcast: what this block explains of the window
        self.fore = nn.Linear(hidden, n_levels)            # forecast: its contribution to each quantile level

    def forward(self, x, m, c):
        # Example (L 6, pool 3): window [3.71 0 4.03 4.26 0 0] -> pooled [2.58 1.42];  input size 2 + 2 + 5 covariates = 9
        px = F.avg_pool1d(x.unsqueeze(1), self.pool).squeeze(1)      # (n, L/pool) coarse view of the window
        pm = F.avg_pool1d(m.unsqueeze(1), self.pool).squeeze(1)      # (n, L/pool) share of real months per bucket
        h = self.mlp(torch.cat([px, pm, c], dim=1))
        return self.back(h), self.fore(h)


class NHiTS(nn.Module):
    """Stacked blocks, coarse to fine. Each block removes what it explained; the forecasts add up."""

    def __init__(self, L, n_cov, n_levels, pools=(3, 1), hidden=64):
        super().__init__()
        self.blocks = nn.ModuleList([Block(L, n_cov, n_levels, p, hidden) for p in pools])

    def forward(self, x, m, c):
        forecast = 0
        for block in self.blocks:
            back, f = block(x, m, c)
            x = (x - back) * m                             # remove what this block explained; padding stays 0
            forecast = forecast + f
        return forecast                                    # (n, n_levels) log1p quantiles


class NHiTSQuantile(WindowedForecaster):
    """N-HiTS trained with several seeds (quantiles averaged). Plugs into Backtester and Evaluator like every other model."""

    def __init__(self, levels, window=6, pools=(3, 1), hidden=64, n_seeds=3, epochs=200, lr=1e-3, patience=20, batch=256,
                 extra=(), extra_stats=None):
        # extra / extra_stats: calendar columns added to the covariates (see WindowBuilder); leave empty for the normal model
        self.n_seeds, self.cfg = n_seeds, dict(pools=pools, hidden=hidden, epochs=epochs, lr=lr, patience=patience, batch=batch)
        super().__init__(levels, self._train, self._predict, WindowBuilder(window, extra=extra, extra_stats=extra_stats), name="nhits")
    def _train_one(self, data, levels, seed):
        # Responsibility: train ONE network. The last 3 months of the training window never give gradients;
        # they only decide when to stop (early stopping) and which epoch's weights to keep.
        # Example (patience 3): validation losses 0.50, 0.46, 0.45, 0.4505, 0.451, 0.452 -> stop, restore epoch 3 (0.45)
        c = self.cfg
        torch.manual_seed(seed)
        X, M, C = make_tensors(data)
        y = torch.tensor(data["y"], dtype=torch.float32)
        tau = torch.tensor(levels, dtype=torch.float32)
        is_train = torch.tensor(data["t"] <= data["t"].max() - 3)
        tr_idx, va_idx = torch.where(is_train)[0], torch.where(~is_train)[0]
        model = NHiTS(X.shape[1], C.shape[1], len(levels), c["pools"], c["hidden"])
        opt = torch.optim.Adam(model.parameters(), lr=c["lr"], weight_decay=1e-5)
        best, best_state, wait = float("inf"), None, 0
        for epoch in range(c["epochs"]):
            model.train()
            order = tr_idx[torch.randperm(len(tr_idx))]
            for i in range(0, len(order), c["batch"]):
                b = order[i:i + c["batch"]]
                opt.zero_grad()
                loss = pinball(model(X[b], M[b], C[b]), y[b], tau)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val = pinball(model(X[va_idx], M[va_idx], C[va_idx]), y[va_idx], tau).item()
            if val < best - 1e-4:
                best, wait, best_state = val, 0, {k: v.clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= c["patience"]:
                    break
        model.load_state_dict(best_state)
        return model.eval()

    def _train(self, data, levels):
        # Responsibility: n_seeds networks from different random starts; a single run varies, the average is steadier.
        return [self._train_one(data, levels, seed) for seed in range(self.n_seeds)]

    @staticmethod
    def _predict(models, data):
        # Responsibility: log1p-scale quantiles = the average over the seeds' outputs.   Example: seeds say 3.1, 3.3, 3.2 -> 3.2
        X, M, C = make_tensors(data)
        with torch.no_grad():
            return np.mean([m(X, M, C).numpy() for m in models], axis=0)