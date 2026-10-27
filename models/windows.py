"""windows.py - cut each series into fixed windows of past months (for N-HiTS and other window models), and plug any model into the backtest."""
import numpy as np

from .quantile import KEYS, ProbForecaster, QuantileForecast
from .transforms import Log1p


class WindowBuilder:
    """For each (series, target month t): the last `window` months before t as log1p demand, a mask of which slots are real, and covariates for month t."""

    def __init__(self, window=6, min_history=1, target="gross_qty", time_scale=41.0, extra=(), extra_stats=None):
        # extra: names of extra calendar columns to append to the covariates (e.g. working_days, vat_change_pp)
        # extra_stats: {column: (mean, std)} used to standardise them: (value - mean) / std. Take them from the CALENDAR table, never from demand.
        # Example: working_days mean 20.4, std 1.8 -> a month with 16 working days becomes (16 - 20.4) / 1.8 = -2.4
        self.window, self.min_history, self.target, self.time_scale = window, min_history, target, time_scale
        self.extra, self.extra_stats = list(extra), extra_stats
        assert not self.extra or all(c in (extra_stats or {}) for c in self.extra), "extra_stats needs (mean, std) for every extra column"
    def arrays(self, rows, history):
        # Responsibility: turn rows into model-ready arrays, using ONLY months before each row's own month.
        # Window:     last L months before t, left-padded with 0 where the series did not exist yet; mask = 1 for real months, 0 for padding.
        # Example:    series born at month 1, t = 3, L = 6, demand [0, 40] -> X = [0,0,0,0,0,3.714], M = [0,0,0,0,1,1]
        #             (the real zero at month 1 has mask 1; the four padding zeros have mask 0)
        # Covariates: [month_sin, month_cos, log1p(age), 1 if age < 3 else 0, t / time_scale]
        # ok:         False when the series has fewer than `min_history` earlier months (age 0 -> no forecast)
        series = {k: (int(g["time_idx"].iloc[0]), g[self.target].to_numpy(float))
                  for k, g in history.sort_values("time_idx").groupby(KEYS, observed=True)}
        n, L = len(rows), self.window
        t = rows["time_idx"].to_numpy()
        keys = list(zip(rows["Site"], rows["Brand"], rows["Range"]))
        X, M, ok = np.zeros((n, L)), np.zeros((n, L)), np.zeros(n, bool)
        for i in range(n):
            start, v = series[keys[i]]
            k = int(t[i] - start)                                  # months of history before t
            if k >= self.min_history:
                past = v[max(0, k - L): k]
                X[i, L - len(past):], M[i, L - len(past):], ok[i] = np.log1p(past), 1.0, True
        age = rows["range_age_months"].to_numpy(float)
        C = np.column_stack([rows["month_sin"].to_numpy(float), rows["month_cos"].to_numpy(float),
                             np.log1p(age), (age < 3).astype(float), t / self.time_scale])
        if self.extra:
            mean = np.array([self.extra_stats[c][0] for c in self.extra], float)
            std = np.array([self.extra_stats[c][1] for c in self.extra], float)
            C = np.column_stack([C, (rows[self.extra].to_numpy(float) - mean) / std])
        y = np.log1p(rows[self.target].to_numpy(float)) if self.target in rows else np.full(n, np.nan)
        return {"X": X, "M": M, "C": C, "y": y, "t": t, "ok": ok}


class WindowedForecaster(ProbForecaster):
    """Adapter: any window model written as two functions becomes a model the backtest and evaluator understand."""

    def __init__(self, levels, train_fn, predict_fn, builder=None, name="windowed"):
        # train_fn(data, levels) -> model        data: X (n,L), M (n,L), C (n,F), y (n,) log1p target, t (n,) month index
        # predict_fn(model, data) -> (n, len(levels)) quantiles on the LOG1P scale, in the order of `levels`
        self.levels, self.train_fn, self.predict_fn = list(levels), train_fn, predict_fn
        self.builder, self.name, self.model = builder or WindowBuilder(), name, None

    @staticmethod
    def _pick(d, keep):
        return {k: v[keep] for k, v in d.items() if k != "ok"}

    def fit(self, train):
        # Responsibility: build training windows (they only look backward, so no leakage) and hand them to your training function.
        d = self.builder.arrays(train, train)
        self.model = self.train_fn(self._pick(d, d["ok"]), self.levels)
        return self

    def predict(self, rows, history):
        # Responsibility: windows from `history` (months < t only) -> your predict function -> units -> sorted, non-negative quantiles.
        # Rows with no earlier month get NaN, like the other models.
        d = self.builder.arrays(rows, history)
        log_q = np.full((len(rows), len(self.levels)), np.nan)
        log_q[d["ok"]] = self.predict_fn(self.model, self._pick(d, d["ok"]))
        return QuantileForecast.from_raw(rows[KEYS + ["time_idx"]], self.levels, Log1p().inverse(log_q))