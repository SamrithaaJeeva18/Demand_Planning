"""quantile.py - what every model returns (QuantileForecast), the rules every model follows (ProbForecaster), the tree models."""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .transforms import Log1p

KEYS = ["Site", "Brand", "Range"]            # one series = one Site x Brand x Range


def level_name(level):
    # Responsibility: column name for a quantile level.
    # Example: 0.1 -> "p10",  0.95 -> "p95"
    return f"p{int(round(level * 100))}"


class QuantileForecast:
    """Forecasts of ONE model for a set of rows: one column per quantile level, plus an optional mean-type point forecast."""

    def __init__(self, keys, levels, values, point=None):
        self.keys = keys                      # DataFrame [Site, Brand, Range, time_idx], one row per forecast
        self.levels = list(levels)            # e.g. [0.1, 0.5, 0.9]
        self.values = np.asarray(values)      # shape (n_rows, n_levels)
        self.point = point                    # optional (n_rows,): a MEAN forecast, for models that make one (e.g. Croston)

    @classmethod
    def from_raw(cls, keys, levels, raw, point=None):
        # Responsibility: make raw model output valid - never negative, quantiles never crossing.
        # Formula:  clip below at 0, then sort each row ascending
        # Example:  raw [210, 190, 380] -> [190, 210, 380];   raw [-5, 20, 60] -> [0, 20, 60]
        return cls(keys, levels, np.sort(np.clip(np.asarray(raw, float), 0, None), axis=1), point)

    def _i(self, level):
        i = int(np.argmin(np.abs(np.asarray(self.levels) - level)))
        if abs(self.levels[i] - level) > 1e-9:
            raise KeyError(f"level {level} not in {self.levels}")
        return i

    def has(self, *levels):
        # Responsibility: does this forecast contain all these levels?   Example: qf.has(0.1, 0.9) -> True / False
        return all(np.min(np.abs(np.asarray(self.levels) - l)) < 1e-9 for l in levels)

    def col(self, level):
        # Responsibility: the values of one level.   Example: qf.col(0.5) -> array of every row's P50
        return self.values[:, self._i(level)]

    def point_forecast(self):
        # Responsibility: the single number used for MAE / WAPE / Bias - the mean if the model has one, else the median.
        return self.point if self.point is not None else self.col(0.5)

    def to_frame(self):
        # Responsibility: a readable table, columns p10, p50, ... with the same index as the rows forecast.
        return pd.DataFrame(self.values, columns=[level_name(l) for l in self.levels], index=self.keys.index)


class ProbForecaster(ABC):
    """Rules every model follows, so any model can be dropped into the same backtest and evaluation."""

    name = "model"

    @abstractmethod
    def fit(self, train):
        """Learn from `train` (rows up to the cutoff). Models that do not learn (zero-shot, naive) just return self."""

    @abstractmethod
    def predict(self, rows, history):
        """Forecast one month ahead for each row. May use `history` ONLY for months before the row's own month.
        Returns a QuantileForecast aligned to `rows`."""


class TreeQuantile(ProbForecaster):
    """Shared logic of the gradient-boosted tree models: one model per quantile level, trained on a transformed target."""

    defaults = {}                             # each subclass sets its own settings

    def __init__(self, levels, features, target="gross_qty", cat_cols=KEYS, transform=None, **params):
        self.levels, self.features, self.target, self.cat_cols = list(levels), list(features), target, list(cat_cols)
        self.transform = transform or Log1p()
        self.params = {**self.defaults, **params}
        self.models, self.cats = {}, {}

    def _make(self, alpha):
        raise NotImplementedError             # subclass: an unfitted regressor for quantile `alpha`

    def _x(self, df):
        # Responsibility: feature matrix whose category lists equal the ones seen in training (the libraries require it).
        x = df[self.features].copy()
        for c in self.cat_cols:
            x[c] = pd.Categorical(df[c].astype(str), categories=self.cats[c])
        return x

    def fit(self, train):
        # Responsibility: train one model per level on transform(target).
        # Formula:  pinball_tau(y, q) = tau*(y - q) if y >= q else (1 - tau)*(q - y)
        # Example:  tau = 0.9, y = 100:  q = 80 -> 0.9*20 = 18;   q = 120 -> 0.1*20 = 2   (too low costs 9x more)
        for c in self.cat_cols:
            self.cats[c] = sorted(train[c].astype(str).unique())
        x, z = self._x(train), self.transform.forward(train[self.target])
        for q in self.levels:
            self.models[q] = self._make(q).fit(x, z)
        return self

    def predict(self, rows, history=None):
        # Responsibility: predict each level, undo the transform, repair crossing/negatives.
        # Example (one row): raw log-scale [4.80, 5.30, 5.89] -> demand [120, 200, 360]
        x = self._x(rows)
        raw = np.column_stack([self.transform.inverse(self.models[q].predict(x)) for q in self.levels])
        return QuantileForecast.from_raw(rows[KEYS + ["time_idx"]], self.levels, raw)


class LGBMQuantile(TreeQuantile):
    """LightGBM: leaf-wise trees, fast, handles NaN and categories natively."""

    name = "lgbm_quantile"
    defaults = {"n_estimators": 300, "random_state": 42, "verbose": -1}

    def _make(self, alpha):
        from lightgbm import LGBMRegressor
        return LGBMRegressor(objective="quantile", alpha=alpha, **self.params)


class XGBQuantile(TreeQuantile):
    """XGBoost: level-wise trees (depth 6), same quantile loss, also handles NaN and categories natively.
    learning_rate 0.1, 300 trees and min_child_weight 20 (= LightGBM's minimum of 20 rows per leaf) keep the comparison
    about the algorithm, not the settings. Without the leaf limit XGBoost may build leaves of a single row and overfit."""

    name = "xgb_quantile"
    defaults = {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 6, "min_child_weight": 20, "random_state": 42,
                "tree_method": "hist", "enable_categorical": True}

    def _make(self, alpha):
        from xgboost import XGBRegressor
        return XGBRegressor(objective="reg:quantileerror", quantile_alpha=alpha, **self.params)


class NaiveLag1(ProbForecaster):
    """The floor every model must beat: next month = last month. Needs the lag_1_gross_qty feature on the rows."""

    name = "naive_lag1"

    def fit(self, train):
        return self

    def predict(self, rows, history=None):
        # Example: last month sold 110 -> forecast 110 (first month of a series has no last month -> NaN)
        p = rows["lag_1_gross_qty"].to_numpy(float)
        return QuantileForecast(rows[KEYS + ["time_idx"]], [0.5], p.reshape(-1, 1), point=p)