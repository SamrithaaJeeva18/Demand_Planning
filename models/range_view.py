"""range_view.py - a Range-level view of one model's forecasts: a table, a summary and small-multiple trend plots."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .uncertainty_range import MonteCarlo, QuantileInterpolator


class RangeView:
    """Adds each Range's series up (Site x Brand x Range -> Range) and shows actual, forecast and uncertainty month by month."""

    def __init__(self, rows, qf, history, target="gross_qty", n_draws=1000, seed=42):
        # Responsibility: keep forecastable rows, draw Monte Carlo samples for each of them, remember every series' actual.
        # Example: Range R1 has 40 series, 3 are new (no forecast) -> 37 are summed; the 3 new ones only appear in `actual_all`.
        ok = ~np.isnan(qf.values).any(axis=1)
        base = rows[["Range", "time_idx", target]].reset_index(drop=True)
        self.levels, self.target, self.history = list(qf.levels), target, history
        self.all_actual = base.groupby(["Range", "time_idx"])[target].sum()
        self.rows, self.y, self.q = base[ok].reset_index(drop=True), base.loc[ok, target].to_numpy(float), qf.values[ok]
        self.samples = MonteCarlo(QuantileInterpolator(self.levels, self.q), n_draws, seed).samples
        self._table = None

    def table(self):
        # Responsibility: one line per Range and validation month, sorted from the smallest Range (by actual volume) to the largest.
        # p10 ... p95: quantiles of the TOTAL = percentiles of the summed Monte Carlo draws (series assumed independent).
        # sum_p10 / sum_p90: the plain sums of every series' own P10 / P90 (series assumed to move together, a wider band).
        # ln_*: log1p of each value. inside_A / inside_B: is the actual inside the P10-P90 band of method A / B?
        # Example: Range R1, month 38: actual 5,200, p10 4,100, p50 5,000, p90 6,300 -> inside_A True; ln_actual = log1p(5200) = 8.56
        if self._table is None:
            out = []
            for (rng, month), idx in self.rows.groupby(["Range", "time_idx"]).indices.items():
                rec = {"Range": rng, "month": month, "n_series": len(idx), "actual": self.y[idx].sum(), "actual_all": self.all_actual[(rng, month)]}
                for lv, v in zip(self.levels, np.percentile(self.samples[idx].sum(axis=0), [100 * l for l in self.levels])):
                    rec[f"p{int(round(lv * 100))}"] = v
                rec["sum_p10"], rec["sum_p90"] = self.q[idx, self.levels.index(0.1)].sum(), self.q[idx, self.levels.index(0.9)].sum()
                out.append(rec)
            t = pd.DataFrame(out)
            t["volume"] = t.groupby("Range")["actual"].transform("sum")
            for c in ["actual"] + [f"p{int(round(lv * 100))}" for lv in self.levels]:
                t[f"ln_{c}"] = np.log1p(t[c])
            t["inside_A"] = (t["actual"] >= t["p10"]) & (t["actual"] <= t["p90"])
            t["inside_B"] = (t["actual"] >= t["sum_p10"]) & (t["actual"] <= t["sum_p90"])
            self._table = t.sort_values(["volume", "Range", "month"]).reset_index(drop=True)
        return self._table

    @property
    def order(self):
        # Responsibility: Range names from smallest to largest volume.   Example: view.order[:8] = the eight smallest Ranges
        return self.table().drop_duplicates("Range")["Range"].tolist()

    def pick(self, k):
        # Responsibility: k Ranges spread evenly from smallest to largest.   Example: 100 Ranges, k = 5 -> positions 0, 25, 50, 74, 99
        pos = np.linspace(0, len(self.order) - 1, k).round().astype(int)
        return [self.order[i] for i in pos]

    def summary(self):
        # Responsibility: one line per Range, smallest to largest.
        # bias % = 100 * (forecast total / actual total - 1).  WAPE = sum |P50 - actual| / sum actual over the months.
        # coverage A / B = share of months whose actual falls inside the P10-P90 band of method A / B (target about 0.80).
        # Example: actual 1,000, P50 total 900 -> bias -10%
        t = self.table().assign(abs_err=lambda d: (d["p50"] - d["actual"]).abs())
        g = t.groupby("Range", sort=False)
        vol = g["actual"].sum()
        return pd.DataFrame({"volume": vol, "P50 total": g["p50"].sum(), "bias %": 100 * (g["p50"].sum() / vol - 1),
                             "WAPE": g["abs_err"].sum() / vol, "coverage A": g["inside_A"].mean(), "coverage B": g["inside_B"].mean(),
                             "series": g["n_series"].max(), "months": g.size()})

    def plot(self, ranges=None, scale="raw", ncols=4):
        # Responsibility: one small chart per Range - grey = history, dot = actual, line = P50, band = P10-P90, dotted = P95.
        # scale "log1p" squeezes big Ranges so small ones stay visible: 5 -> 1.79, 500 -> 6.22, 50,000 -> 10.82
        f = np.log1p if scale == "log1p" else (lambda v: np.asarray(v, float))
        ranges = list(ranges) if ranges is not None else self.pick(12)
        t, h = self.table(), self.history
        cut = t["month"].min()
        nrows = int(np.ceil(len(ranges) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows), squeeze=False)
        for ax, r in zip(axes.ravel(), ranges):
            d = t[t["Range"] == r].sort_values("month")
            past = h[(h["Range"] == r) & (h["time_idx"] < cut)].groupby("time_idx")[self.target].sum()
            ax.plot(past.index, f(past.values), color="gray", lw=1, label="history")
            ax.fill_between(d["month"], f(d["p10"]), f(d["p90"]), alpha=0.3, label="P10-P90")
            if "p95" in d:
                ax.plot(d["month"], f(d["p95"]), ":", lw=1, label="P95")
            ax.plot(d["month"], f(d["p50"]), label="P50")
            ax.plot(d["month"], f(d["actual"]), "ko", ms=3, label="actual")
            ax.axvline(cut - 0.5, color="black", lw=0.5, ls="--")
            ax.set_title(f"{r}  volume {d['volume'].iloc[0]:,.0f}  in band {d['inside_A'].mean():.0%}", fontsize=8)
        for ax in axes.ravel()[len(ranges):]:
            ax.axis("off")
        axes[0, 0].legend(fontsize=6)
        fig.tight_layout()
        return fig