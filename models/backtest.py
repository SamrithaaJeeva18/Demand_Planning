"""backtest.py - the fold calendar, a runner that retrains per fold, and a guard that proves no model sees the answer."""
import numpy as np


class RollingOrigin:
    """Fold calendar: train on months <= cutoff, validate on the next `horizon` months, then move the cutoff forward."""

    def __init__(self, last_month, horizon=6, n_folds=4):
        # Formula:  cutoff_k = last_month - horizon * k  for k = n_folds ... 1
        # Example:  last_month 41, horizon 6, 4 folds -> cutoffs 17, 23, 29, 35 -> validate months 18-23, 24-29, 30-35, 36-41
        self.horizon = horizon
        self.cutoffs = [last_month - horizon * k for k in range(n_folds, 0, -1)]

    def split(self, panel, cutoff):
        # Responsibility: (train, validation) rows for one fold.   Example: cutoff 35 -> train months 0-35, validate 36-41
        t = panel["time_idx"]
        return panel[t <= cutoff], panel[(t > cutoff) & (t <= cutoff + self.horizon)]


class Backtester:
    """Runs a model over every fold: fit on the past, forecast the next months. Keeps each fold's fitted model in `fitted`."""

    def __init__(self, panel, folds):
        self.panel, self.folds, self.fitted = panel, folds, {}

    def run(self, make_model):
        # Responsibility: for each cutoff -> {cutoff: (validation rows, QuantileForecast)}.
        # `make_model` builds a FRESH model per fold, e.g. lambda: LGBMQuantile(levels, features).
        # Afterwards self.fitted[cutoff] is that fold's model (for the leakage guard, feature importance, ...).
        out = {}
        for c in self.folds.cutoffs:
            train, val = self.folds.split(self.panel, c)
            model = make_model().fit(train)
            self.fitted[c] = model
            out[c] = (val, model.predict(val, self.panel))
        return out


def assert_no_leakage(model, panel, rows, target="gross_qty", per_month=20, rtol=1e-4, atol=1e-3):
    # Responsibility: prove a model cannot see the answer for the month it forecasts.
    # Method:   for each forecast month t, set the actual of month t AND LATER to 1e9 (history and rows), forecast again;
    #           the forecasts for month t must not change beyond float rounding (atol = 0.001 units).
    # Example:  R1's July actual 110 -> 1e9: July's forecast must stay identical.
    rows = rows.groupby("time_idx", group_keys=False).head(per_month)
    base = model.predict(rows, panel).values
    for t in np.unique(rows["time_idx"]):
        m = (rows["time_idx"] == t).to_numpy()
        p2 = panel.copy()
        p2.loc[p2["time_idx"] >= t, target] = 1e9
        r2 = rows[m].copy()
        r2[target] = 1e9
        again = model.predict(r2, p2).values
        diff = np.nanmax(np.abs(base[m] - again)) if np.isfinite(base[m]).any() else 0.0
        assert np.allclose(base[m], again, rtol=rtol, atol=atol, equal_nan=True), f"LEAK: month {t}, largest difference {diff:.4g} units"