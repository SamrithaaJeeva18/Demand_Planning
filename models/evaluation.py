"""evaluation.py - every metric from the teaching (Metrics) and a fair side-by-side scorer (Evaluator)."""
import numpy as np
import pandas as pd


class Metrics:
    """One small static function per metric. y = actual, p = forecast. Numbers use the 8-row teaching example."""

    @staticmethod
    def mae(y, p):
        # Typical miss in units.   Formula: mean |p - y|.   Example: 56.625
        return np.mean(np.abs(p - y))

    @staticmethod
    def rmse(y, p):
        # Like MAE but big misses count more.   Formula: sqrt(mean (p - y)^2).   Example: 116.9
        return np.sqrt(np.mean((p - y) ** 2))

    @staticmethod
    def wape(y, p):
        # Total miss as a share of total demand.   Formula: sum|p - y| / sum|y|.   Example: 453 / 1455 = 0.311
        d = np.abs(y).sum()
        return np.abs(p - y).sum() / d if d > 0 else np.nan          # no demand in the group -> undefined, not a crash

    @staticmethod
    def bias(y, p):
        # Direction of the error, negative = under-forecasting.   Formula: mean (p - y).   Example: -43.375
        return np.mean(p - y)

    @staticmethod
    def pinball(y, q, tau):
        # Asymmetric penalty for a quantile forecast q at level tau.   Formula: tau*(y-q) if y>=q else (1-tau)*(q-y)
        # Example (8 rows): tau 0.1 -> 9.19;  tau 0.5 -> 28.31;  tau 0.9 -> 21.19
        d = y - q
        return np.mean(np.maximum(tau * d, (tau - 1) * d))

    @staticmethod
    def coverage(y, lo, hi):
        # Share of actuals inside [lo, hi].   Formula: mean(lo <= y <= hi).   Example: 7 of 8 = 0.875 (target 0.80 for P10-P90)
        return np.mean((y >= lo) & (y <= hi))

    @staticmethod
    def width(lo, hi):
        # Sharpness: narrower is better only if coverage holds.   Formula: mean (hi - lo).   Example: 128.75
        return np.mean(hi - lo)

    @staticmethod
    def interval_score(y, lo, hi, alpha=0.2):
        # Width plus a penalty of 2/alpha = 10x the distance outside.   Formula: (hi-lo) + 10*max(lo-y,0) + 10*max(y-hi,0)
        # Example: row with actual 700 vs band [250, 560] -> 310 + 10*140 = 1710;  mean of the 8 rows = 303.75
        return np.mean((hi - lo) + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0))

    @staticmethod
    def crps(y, samples):
        # Whole-curve score per row, in units.   Formula: E|X - y| - 0.5 * E|X - X'|   (X, X' = two draws)
        # Example: actual 120 against the R1 curve -> about 54.2
        n = samples.shape[1]
        s = np.sort(samples, axis=1)
        i = np.arange(1, n + 1)
        e_xy = np.mean(np.abs(s - y[:, None]), axis=1)
        e_xx = (2.0 / n ** 2) * (s * (2 * i - n - 1)).sum(axis=1)      # E|X-X'| from sorted draws
        return e_xy - 0.5 * e_xx

    @staticmethod
    def pit(y, samples, seed=0):
        # Where the actual lands inside its own forecast (0 = below every draw, 1 = above every draw); flat histogram = calibrated.
        # Formula: share of draws below y; ties (many zeros) are spread at random.   Example: actual 120 vs R1 curve -> 0.099
        rng = np.random.default_rng(seed)
        lower, upper = (samples < y[:, None]).mean(axis=1), (samples <= y[:, None]).mean(axis=1)
        return lower + rng.random(len(y)) * (upper - lower)


def age_bucket(age):
    # Responsibility: label each row new / middle / established by months of history.
    # Example: age 1 -> "0-2 (new)";  age 8 -> "3-11";  age 20 -> "12+ (established)"
    return pd.cut(pd.Series(age), [-1, 2, 11, 10 ** 6], labels=["0-2 (new)", "3-11", "12+ (established)"]).astype(str).to_numpy()


def size_decile(actual):
    # Responsibility: decile 0-9 of NON-ZERO actuals (zeros get -1), so error can be read by demand size.
    # Example: the smallest tenth of sales -> 0, the biggest tenth -> 9, a month with no sales -> -1
    a = pd.Series(actual)
    out = pd.Series(-1, index=a.index)
    nz = a > 0
    out[nz] = pd.qcut(a[nz].rank(method="first"), 10, labels=False)
    return out.to_numpy()


class Evaluator:
    """Scores any number of models on the SAME rows, so comparisons are fair."""

    def __init__(self, rows, actual):
        self.index = rows.index
        self.actual = np.asarray(actual, float)
        self.age = rows["range_age_months"].to_numpy(float)

    def _ok(self, forecasts, eligible):
        ok = np.ones(len(self.actual), bool) if eligible is None else np.asarray(eligible, bool).copy()
        for qf in forecasts.values():
            assert len(qf.values) == len(self.actual), "forecast and rows have different lengths"
            ok &= ~np.isnan(qf.point_forecast())
        return ok

    def table(self, forecasts, eligible=None):
        # Responsibility: one row per model, all scored on rows EVERY model can forecast.
        # Columns: MAE, WAPE, Bias (on each model's point forecast), WAPE for mature / young series,
        #          and, when the model has P10/P50/P90: coverage, mean pinball, interval score.
        ok = self._ok(forecasts, eligible)
        y, age = self.actual[ok], self.age[ok]
        rows = {}
        for name, qf in forecasts.items():
            p = qf.point_forecast()[ok]
            r = {"MAE": Metrics.mae(y, p), "RMSE": Metrics.rmse(y, p), "WAPE": Metrics.wape(y, p), "Bias": Metrics.bias(y, p),
                 "WAPE mature (age>=12)": Metrics.wape(y[age >= 12], p[age >= 12]),
                 "WAPE young (age<12)": Metrics.wape(y[age < 12], p[age < 12])}
            if qf.has(0.1, 0.5, 0.9):
                lo, hi = qf.col(0.1)[ok], qf.col(0.9)[ok]
                r["coverage [P10,P90]"] = Metrics.coverage(y, lo, hi)
                r["mean pinball"] = np.mean([Metrics.pinball(y, qf.col(t)[ok], t) for t in (0.1, 0.5, 0.9)])
                r["interval score"] = Metrics.interval_score(y, lo, hi)
            rows[name] = r
        out = pd.DataFrame(rows).T
        out["n_rows"] = int(ok.sum())
        return out

    def reliability(self, qf):
        # Responsibility: for each level, what share of actuals fall below it? Calibrated = nominal sits between the two columns
        # (strict "<" and inclusive "<=" both shown, because many actuals equal 0 and tie with a quantile of 0).
        y, ok = self.actual, ~np.isnan(qf.values).any(axis=1)
        rows = {l: {"observed_below_strict": np.mean(y[ok] < qf.col(l)[ok]),
                    "observed_below_inclusive": np.mean(y[ok] <= qf.col(l)[ok])} for l in qf.levels}
        return pd.DataFrame(rows).T.rename_axis("nominal")

    def by_segment(self, qf, labels):
        # Responsibility: error and coverage inside each segment (decile, age bucket, ...).
        # Example: labels = age_bucket(rows.range_age_months) -> one line for "0-2 (new)", "3-11", "12+ (established)"
        labels, ok = np.asarray(labels), ~np.isnan(qf.point_forecast())
        out = {}
        for lab in pd.unique(labels[ok]):
            m = ok & (labels == lab)
            r = {"n_rows": int(m.sum()), "MAE": Metrics.mae(self.actual[m], qf.point_forecast()[m]),
                 "WAPE": Metrics.wape(self.actual[m], qf.point_forecast()[m]) if self.actual[m].sum() > 0 else np.nan}
            if qf.has(0.1, 0.9):
                r["coverage [P10,P90]"] = Metrics.coverage(self.actual[m], qf.col(0.1)[m], qf.col(0.9)[m])
            out[lab] = r
        return pd.DataFrame(out).T.sort_index()