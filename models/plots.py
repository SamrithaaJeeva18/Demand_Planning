"""plots.py - the standard diagnostic plots and checks for any model's QuantileForecast."""
import matplotlib.pyplot as plt
import numpy as np

from .evaluation import Metrics, size_decile
from .uncertainty_range import MonteCarlo, QuantileInterpolator, TwoPieceCurve


class ForecastPlots:
    """Plots for ONE model on one set of rows. Rows the model could not forecast (NaN) are dropped once, here."""

    def __init__(self, rows, qf, name="model", target="gross_qty", n_draws=1000, seed=42, sampler="interp"):
        # Responsibility: keep only forecastable rows and draw Monte Carlo samples.
        # sampler "interp":    straight lines between ALL the model's quantile levels (matches the model by construction;
        #                      tails beyond the outer levels are clamped, so totals can come out slightly narrow)
        # sampler "two_piece": a bell curve fitted to P10 / P50 / P90 only (smooth tails, but may disagree with the model's P80 / P95)
        # Example: 3,000 rows, 120 of them age 0 -> 2,880 kept; samples shape (2880, 1000)
        self.ok = ~np.isnan(qf.values).any(axis=1)
        self.name, self.levels, self.q = name, qf.levels, qf.values[self.ok]
        self.month = rows["time_idx"].to_numpy()[self.ok]
        self.y = rows[target].to_numpy(float)[self.ok]
        self.p10, self.p50, self.p90 = qf.col(0.1)[self.ok], qf.col(0.5)[self.ok], qf.col(0.9)[self.ok]
        self.curve = TwoPieceCurve(self.p10, self.p50, self.p90)
        source = QuantileInterpolator(self.levels, self.q) if sampler == "interp" else self.curve
        self.mc, self.n_draws, self.seed = MonteCarlo(source, n_draws, seed), n_draws, seed

    # ---------- numbers ----------
    def shape_check(self):
        # Responsibility: P80 and P95 were NOT used to fit the curve, so they test its shape against the model's own P80 / P95.
        # Formula:  gap % = 100 * (curve total / model total - 1), totals summed over all rows
        # Example:  model P95 total 52,000, curve P95 total 49,400 -> gap -5.0% (curve too narrow in the tail)
        import pandas as pd
        out = {}
        for lv in (0.8, 0.95):
            if lv in self.levels:
                model_q = self.q[:, self.levels.index(lv)]
                curve_q = MonteCarlo(self.curve, self.n_draws, self.seed).quantiles([lv])[:, 0]
                out[f"P{int(lv * 100)}"] = {"model total": model_q.sum(), "curve total": curve_q.sum(),
                                           "gap %": 100 * (curve_q.sum() / model_q.sum() - 1),
                                           "median |row gap|": float(np.median(np.abs(curve_q - model_q)))}
        return pd.DataFrame(out).T

    def scores(self):
        # Responsibility: two whole-curve numbers. CRPS = mean miss of the full forecast in units (lower is better);
        # PIT deviation = largest gap between any of 10 histogram bars and the ideal 10% (0 = perfectly calibrated).
        # Example: bars 0.05 ... 0.22 -> deviation 0.12
        pit = Metrics.pit(self.y, self.mc.samples)
        bars = np.histogram(pit, bins=10, range=(0, 1))[0] / len(pit)
        return {"mean CRPS": float(Metrics.crps(self.y, self.mc.samples).mean()), "PIT max deviation": float(np.abs(bars - 0.1).max())}

    # ---------- plots ----------
    @staticmethod
    def _ax(ax, size=(7, 4)):
        return ax if ax is not None else plt.subplots(figsize=size)[1]

    def fan(self, ax=None, mask=None):
        # Responsibility: monthly TOTAL demand, actual vs forecast, with two bands.
        # Band A: P10 / P90 of the summed draws (rows assumed independent). Band B: sum of the rows' P10s / P90s (rows move together).
        # Example: two identical rows with P90 = 360 each -> band A top about 645, band B top 720
        ax = self._ax(ax, (8, 4))
        keep = np.ones(len(self.y), bool) if mask is None else np.asarray(mask)[self.ok]
        months = np.unique(self.month[keep])
        rec = []
        for t in months:
            i = np.where(keep & (self.month == t))[0]
            tot = self.mc.samples[i].sum(axis=0)
            rec.append((self.y[i].sum(), self.p50[i].sum(), np.percentile(tot, 10), np.percentile(tot, 90), self.p10[i].sum(), self.p90[i].sum()))
        act, mid, a_lo, a_hi, b_lo, b_hi = np.array(rec).T
        ax.fill_between(months, b_lo, b_hi, alpha=0.15, label="band B: sum of P10 / P90")
        ax.fill_between(months, a_lo, a_hi, alpha=0.35, label="band A: Monte Carlo sum")
        ax.plot(months, mid, label="sum of P50"); ax.plot(months, act, "o-", color="black", label="actual")
        ax.set_title(f"{self.name}: monthly total (A caught {int(((act >= a_lo) & (act <= a_hi)).sum())}/{len(months)}, B caught {int(((act >= b_lo) & (act <= b_hi)).sum())}/{len(months)})")
        ax.set_xlabel("month index"); ax.legend(fontsize=8)

    def predicted_vs_actual(self, ax=None):
        # Responsibility: does P50 follow the actual? Points on the diagonal = perfect; symlog axes so 5 and 5,000 both show.
        ax = self._ax(ax)
        ax.scatter(self.p50, self.y, s=6, alpha=0.3)
        top = max(self.p50.max(), self.y.max())
        ax.plot([0, top], [0, top], color="black", lw=1)
        ax.set_xscale("symlog", linthresh=1); ax.set_yscale("symlog", linthresh=1)
        ax.set_xlabel("P50 forecast"); ax.set_ylabel("actual"); ax.set_title(f"{self.name}: predicted vs actual")

    def residuals(self, ax=None):
        # Responsibility: log-scale error. Formula: log1p(actual) - log1p(P50). Example: actual 90, P50 23.5 -> 4.51 - 3.20 = +1.31 (under-forecast)
        ax = self._ax(ax)
        r = np.log1p(self.y) - np.log1p(self.p50)
        ax.hist(r, bins=50); ax.axvline(0, color="black", lw=1)
        ax.set_title(f"{self.name}: log residuals (median {np.median(r):+.2f})"); ax.set_xlabel("log1p(actual) - log1p(P50)")

    def mae_by_decile(self, ax=None):
        # Responsibility: error by demand size (-1 = months with no sales, 0 = smallest sales ... 9 = biggest), with each group's share of volume.
        # Example: decile 9 holds 72% of all units, so its MAE matters most for stock.
        ax = self._ax(ax)
        g = size_decile(self.y)
        keys = sorted(set(g))
        mae = [np.mean(np.abs(self.y[g == k] - self.p50[g == k])) for k in keys]
        ax.bar([str(k) for k in keys], mae)
        for i, k in enumerate(keys):
            ax.text(i, mae[i], f"{100 * self.y[g == k].sum() / self.y.sum():.0f}%", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"{self.name}: MAE by size decile (label = share of volume)"); ax.set_xlabel("decile")

    def reliability(self, ax=None):
        # Responsibility: for each level, what share of actuals fall at or below it? A calibrated model hugs the diagonal.
        # Strict (<) and inclusive (<=) both shown because many actuals are 0 and tie with a quantile of 0.
        # Example: level 0.9, 87% of actuals <= P90 -> observed 0.87 vs nominal 0.90
        ax = self._ax(ax)
        incl = [np.mean(self.y <= self.q[:, j]) for j in range(len(self.levels))]
        strict = [np.mean(self.y < self.q[:, j]) for j in range(len(self.levels))]
        ax.plot([0, 1], [0, 1], color="black", lw=1)
        ax.plot(self.levels, incl, "o-", label="at or below"); ax.plot(self.levels, strict, "s--", label="strictly below")
        ax.set_xlabel("nominal level"); ax.set_ylabel("observed share"); ax.set_title(f"{self.name}: reliability"); ax.legend(fontsize=8)

    def pit(self, ax=None):
        # Responsibility: where each actual lands inside its own forecast. Flat bars at 10% = calibrated;
        # U-shape = too narrow; hill-shape = too wide; lean = biased.
        ax = self._ax(ax)
        p = Metrics.pit(self.y, self.mc.samples)
        ax.hist(p, bins=10, range=(0, 1), weights=np.ones(len(p)) / len(p)); ax.axhline(0.1, color="black", lw=1)
        ax.set_title(f"{self.name}: PIT histogram"); ax.set_xlabel("PIT value")

    def coverage_by_segment(self, labels, ax=None):
        # Responsibility: is the P10-P90 band honest inside each group (age bucket, demand pattern, ...)? Target line 0.80.
        # Formula: coverage = share of actuals with P10 <= actual <= P90. Example: 7 of 8 -> 0.875
        ax = self._ax(ax)
        lab = np.asarray(labels)[self.ok]
        keys = sorted(set(lab))
        cov = [np.mean((self.y[lab == k] >= self.p10[lab == k]) & (self.y[lab == k] <= self.p90[lab == k])) for k in keys]
        ax.bar(keys, cov); ax.axhline(0.8, color="black", lw=1, ls="--")
        for i, k in enumerate(keys):
            ax.text(i, cov[i], f"n={int((lab == k).sum())}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, 1.05); ax.set_title(f"{self.name}: P10-P90 coverage by segment")

    def width_vs_error(self, ax=None):
        # Responsibility: when the band is wide, is the miss really bigger? A useful band widens where errors are big.
        # Formula: x = P90 - P10, y = |actual - P50|; rank correlation near +1 is good, near 0 means the width carries no information.
        ax = self._ax(ax)
        w, e = self.p90 - self.p10, np.abs(self.y - self.p50)
        corr = np.corrcoef(w.argsort().argsort(), e.argsort().argsort())[0, 1]
        ax.scatter(w, e, s=6, alpha=0.3)
        ax.set_xscale("symlog", linthresh=1); ax.set_yscale("symlog", linthresh=1)
        ax.set_title(f"{self.name}: width vs error (rank corr {corr:.2f})"); ax.set_xlabel("P90 - P10"); ax.set_ylabel("|actual - P50|")

    def grid(self, labels=None):
        # Responsibility: all eight single-model plots on one figure (the ninth, stability over folds, needs several folds).
        fig, axes = plt.subplots(4, 2, figsize=(13, 17))
        self.fan(axes[0, 0]); self.predicted_vs_actual(axes[0, 1]); self.residuals(axes[1, 0]); self.mae_by_decile(axes[1, 1])
        self.reliability(axes[2, 0]); self.pit(axes[2, 1])
        if labels is not None:
            self.coverage_by_segment(labels, axes[3, 0])
        self.width_vs_error(axes[3, 1])
        fig.tight_layout()
        return fig

    @staticmethod
    def folds(tables, model, metrics=("WAPE", "RMSE", "coverage [P10,P90]", "mean pinball")):
        # Responsibility: stability over time. `tables` = {fold name: Evaluator.table(...)}; one small panel per metric.
        # Example: coverage 0.78, 0.81, 0.62, 0.79 over four folds -> the third fold is the problem, not the model overall.
        fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.2))
        for ax, m in zip(axes, metrics):
            vals = [t.loc[model, m] for t in tables.values()]
            ax.plot(range(len(vals)), vals, "o-"); ax.set_xticks(range(len(vals))); ax.set_xticklabels(list(tables), rotation=30, fontsize=7); ax.set_title(m, fontsize=9)
            if "coverage" in m:
                ax.axhline(0.8, color="black", lw=1, ls="--")
        fig.tight_layout()
        return fig