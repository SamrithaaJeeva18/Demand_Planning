"""Every worked example from the teaching, as an assert. Run:  uv run python tests/test_examples.py"""
import numpy as np
import pandas as pd

from models.transforms import Log1p, Identity
from models.quantile import QuantileForecast, LGBMQuantile, NaiveLag1, KEYS
from models.uncertainty_range import TwoPieceCurve, QuantileInterpolator, MonteCarlo
from models.evaluation import Metrics, Evaluator, age_bucket, size_decile
from models.backtest import RollingOrigin, Backtester, assert_no_leakage
# from dforecast.models_fm import Chronos2Zero  # retired; forecasting-model example removed


def close(got, want, tol=0.01):
    assert abs(got - want) <= tol * max(1.0, abs(want)), f"got {got}, expected {want}"


# ---- transforms ----
close(Log1p().forward(200), 5.3033, 1e-3)
close(Log1p().inverse(5.3033), 200, 1e-3)
y = np.array([2, 5, 9.0])
close(Log1p().inverse(np.median(Log1p().forward(y))), 5.0, 1e-6)          # quantiles survive log1p

# ---- QuantileForecast repair ----
qf = QuantileForecast.from_raw(pd.DataFrame(index=[0, 1]), [0.1, 0.5, 0.9], [[210, 190, 380], [-5, 20, 60]])
assert qf.values.tolist() == [[190, 210, 380], [0, 20, 60]]

# ---- two-piece curve + Monte Carlo (R1 example: P10=120, P50=200, P90=360) ----
curve = TwoPieceCurve(np.array([120.0]), np.array([200.0]), np.array([360.0]))
close(curve.mu[0], 5.3033, 1e-3); close(curve.sig_l[0], 0.396, 1e-3); close(curve.sig_r[0], 0.457, 1e-3)
close(curve.from_z(np.array([[0.0]]))[0, 0], 200, 1e-3)
close(curve.from_z(np.array([[1.2816]]))[0, 0], 360, 1e-3)
close(curve.from_z(np.array([[-1.2816]]))[0, 0], 120, 1e-3)
mc = MonteCarlo(curve, n_draws=200_000, seed=7)
for got, want in [(mc.quantiles([0.1, 0.5, 0.9]), [120, 200, 360]), (mc.quantiles([0.8]), [294]), (mc.quantiles([0.95]), [425])]:
    assert np.allclose(got[0], want, rtol=0.03), (got, want)
close(mc.mean()[0], 225, 0.03); close(mc.prob_above(300)[0], 0.188, 0.03)

# ---- quantile interpolation example: u = 0.62 -> 99.2 ----
interp = QuantileInterpolator([0.1, 0.25, 0.5, 0.75, 0.9], np.array([[20, 45, 80, 120, 150.0]]))
close(interp.from_u(np.array([[0.62]]))[0, 0], 99.2, 1e-6)
close(interp.from_u(np.array([[0.05]]))[0, 0], 20, 1e-6)                    # below the outer level: nearest value
close(interp.from_u(np.array([[0.99]]))[0, 0], 150, 1e-6)
d = MonteCarlo(interp, n_draws=100_000, seed=1).quantiles([0.25, 0.5, 0.75])[0]
assert np.allclose(d, [45, 80, 120], rtol=0.03), d

# ---- aggregation: quantiles do not add (two identical series, P90 = 360) ----
rng = np.random.default_rng(3)
z1, z2 = rng.standard_normal((1, 200_000)), rng.standard_normal((1, 200_000))
z2c = 0.8 * z1 + np.sqrt(1 - 0.64) * rng.standard_normal((1, 200_000))
a, b, bc = curve.from_z(z1)[0], curve.from_z(z2)[0], curve.from_z(z2c)[0]
close(np.percentile(a, 90), 360, 0.03)
close(np.percentile(a + b, 90), 645, 0.03); close(np.percentile(a + bc, 90), 705, 0.03); close(np.percentile(a + a, 90), 720, 0.03)

# ---- metrics: the 8-row example ----
y = np.array([40, 120, 15, 300, 80, 0, 700, 200.0]); p10 = np.array([25, 90, 5, 150, 60, 0, 250, 140.0])
p50 = np.array([45, 130, 20, 220, 95, 8, 380, 210.0]); p90 = np.array([80, 190, 50, 350, 150, 40, 560, 330.0])
close(Metrics.mae(y, p50), 56.625); close(Metrics.rmse(y, p50), 116.9); close(Metrics.wape(y, p50), 0.311); close(Metrics.bias(y, p50), -43.375)
close(Metrics.pinball(y, p10, 0.1), 9.19); close(Metrics.pinball(y, p50, 0.5), 28.31); close(Metrics.pinball(y, p90, 0.9), 21.19)
close(Metrics.coverage(y, p10, p90), 0.875); close(Metrics.width(p10, p90), 128.75); close(Metrics.interval_score(y, p10, p90), 303.75)
s = curve.draw(20_000, 11)
close(Metrics.crps(np.array([120.0]), s)[0], 54.2, 0.04); close(Metrics.pit(np.array([120.0]), s)[0], 0.099, 0.2)

# ---- helpers ----
assert age_bucket([1, 8, 20]).tolist() == ["0-2 (new)", "3-11", "12+ (established)"]
assert size_decile([0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]).tolist()[0] == -1

# ---- rolling origin: 42 months, horizon 6, 4 folds ----
assert RollingOrigin(41, 6, 4).cutoffs == [17, 23, 29, 35]

# ---- a tiny panel to test models, evaluator, backtest and the leakage guard ----
rows = []
for s_i in range(6):
    for t in range(0, 24):
        rows.append(("S%d" % (s_i % 3), "B1", "R%d" % s_i, t, float((t * 3 + s_i * 5) % 17 + 1)))
panel = pd.DataFrame(rows, columns=KEYS + ["time_idx", "gross_qty"])
g = panel.groupby(KEYS)["gross_qty"]
panel["lag_1_gross_qty"] = g.shift(1)
panel["range_age_months"] = panel["time_idx"] - panel.groupby(KEYS)["time_idx"].transform("min")
panel["month_sin"] = np.sin(2 * np.pi * panel["time_idx"] / 12)


levels = [0.1, 0.5, 0.9]
bt = Backtester(panel, RollingOrigin(23, horizon=3, n_folds=2))
res = bt.run(lambda: LGBMQuantile(levels, ["time_idx", "month_sin", "lag_1_gross_qty", "range_age_months"] + KEYS, n_estimators=20))
assert len(res) == 2
val, qf = list(res.values())[-1]
assert (np.diff(qf.values, axis=1) >= 0).all() and (qf.values >= 0).all()
assert_no_leakage(
    LGBMQuantile(levels, ["time_idx", "month_sin", "lag_1_gross_qty", "range_age_months"] + KEYS, n_estimators=20)
    .fit(panel[panel["time_idx"] <= 20]),
    panel,
    val,
)
tab = Evaluator(val, val["gross_qty"].to_numpy()).table({"naive": NaiveLag1().predict(val), "lgbm": qf})
assert set(tab.index) == {"naive", "lgbm"} and tab["n_rows"].nunique() == 1

# ---- windows (N-HiTS input): the demo from the teaching ----
from models.windows import WindowBuilder, WindowedForecaster

demo = {"A": {m: v for m, v in zip(range(1, 11), [0, 40, 0, 55, 70, 0, 0, 90, 65, 80])},
        "D": {m: v for m, v in zip(range(6, 11), [20, 0, 35, 0, 30])}}
dr = []
for s, ser in demo.items():
    start = min(ser)
    for m, v in ser.items():
        dr.append((s, "B1", "R1", m, float(v), np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12), m - start))
dpanel = pd.DataFrame(dr, columns=KEYS + ["time_idx", "gross_qty", "month_sin", "month_cos", "range_age_months"])
wb = WindowBuilder(window=6, time_scale=10.0)
pick = lambda s, t: dpanel[(dpanel["Site"] == s) & (dpanel["time_idx"] == t)]

d = wb.arrays(pick("A", 3), dpanel)                                   # 2 months of history: 4 slots of padding
assert np.allclose(d["X"][0], [0, 0, 0, 0, 0, 3.7136], atol=1e-3) and d["M"][0].tolist() == [0, 0, 0, 0, 1, 1]
assert d["ok"][0] and abs(d["y"][0] - 0.0) < 1e-9
assert np.allclose(d["C"][0], [1.0, 0.0, 1.0986, 1.0, 0.3], atol=1e-3)                # sin, cos, log1p(age 2), new flag, t/10
d = wb.arrays(pick("A", 8), dpanel)                                   # a full window
assert np.allclose(d["X"][0], [3.7136, 0, 4.0254, 4.2627, 0, 0], atol=1e-3) and d["M"][0].tolist() == [1] * 6
close(d["y"][0], 4.5109, 1e-3)
d = wb.arrays(pick("D", 8), dpanel)                                   # young series: the zero at month 7 is REAL (mask 1)
assert np.allclose(d["X"][0], [0, 0, 0, 0, 3.0445, 0], atol=1e-3) and d["M"][0].tolist() == [0, 0, 0, 0, 1, 1]
assert not wb.arrays(pick("D", 6), dpanel)["ok"][0]                   # age 0: no history, no forecast
assert WindowBuilder(window=12).arrays(pick("A", 8), dpanel)["X"].shape == (1, 12)

# ---- the adapter: a baseline (quantiles of the real months in the window) must run, align, and not leak ----
def _train(data, lv): return None
def _predict(model, data):
    return np.array([np.quantile(x[m > 0], levels) for x, m in zip(data["X"], data["M"])])

wf = WindowedForecaster(levels, _train, _predict, WindowBuilder(window=6), name="window_quantiles")
vw = dpanel[dpanel["time_idx"] >= 6]
qw = wf.fit(dpanel[dpanel["time_idx"] <= 8]).predict(vw, dpanel)
assert qw.values.shape == (len(vw), 3) and np.isnan(qw.values[(vw["Site"] == "D").to_numpy() & (vw["time_idx"] == 6).to_numpy()]).all()
good = ~np.isnan(qw.values).any(axis=1)
assert (np.diff(qw.values[good], axis=1) >= 0).all() and (qw.values[good] >= 0).all()
assert_no_leakage(wf, dpanel, vw)

print("ALL EXAMPLES AND CHECKS PASSED")