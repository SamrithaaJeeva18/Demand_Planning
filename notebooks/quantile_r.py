import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from lightgbm import LGBMRegressor

    # ---- constants: the same numbers used in the teaching examples ----
    Z90 = 1.2816                        # a normal curve's 90th percentile sits 1.2816 sigma from its centre
    FIT_Q = [0.1, 0.5, 0.9]             # quantiles the curve is FITTED from
    CHECK_Q = [0.8, 0.95]               # trained too, but NOT used for fitting - they test the curve's shape
    ALL_Q = sorted(FIT_Q + CHECK_Q)     # [0.1, 0.5, 0.8, 0.9, 0.95]
    N_DRAWS = 20000                      # Monte Carlo draws per row
    SEED = 42
    VAL_MONTHS = 6                      # last 6 months held out, same split as the baseline models
    TARGET = "gross_qty"
    CAT_COLS = ["Site", "Brand", "Range"]
    FEATURES = ["year", "month_sin", "month_cos", "time_idx", "range_age_months",
                "lag_1_gross_qty", "rolling_mean_3", "rolling_mean_6", "lag_12_gross_qty",
                "return_rate_prev_month", "Site", "Brand", "Range"]
    return (
        ALL_Q,
        CAT_COLS,
        FEATURES,
        FIT_Q,
        LGBMRegressor,
        N_DRAWS,
        Path,
        SEED,
        TARGET,
        VAL_MONTHS,
        Z90,
        np,
        pd,
        plt,
    )


@app.cell
def _(CAT_COLS, Path, TARGET, VAL_MONTHS, pd):
    # ---- load the saved dataset and rebuild the SAME time split used for the baselines ----
    data_q = pd.read_pickle(Path("data/processed/training_data.pkl"))
    data_q = data_q.dropna(subset=[TARGET])

    cutoff_q = data_q["time_idx"].max() - VAL_MONTHS
    train_q = data_q[data_q["time_idx"] <= cutoff_q].copy()
    val_q = data_q[data_q["time_idx"] > cutoff_q].copy()

    # make val use exactly the same category list as train (LightGBM needs this)
    for _c in CAT_COLS:
        train_q[_c] = train_q[_c].astype(str).astype("category")
        val_q[_c] = pd.Categorical(val_q[_c].astype(str), categories=train_q[_c].cat.categories)

    print(f"train rows: {len(train_q):,} | val rows: {len(val_q):,}   (baseline runs had 3,241 val rows)")
    print(f"share of zero-demand rows in val: {(val_q[TARGET] == 0).mean():.1%}   (about 52% expected)")
    return data_q, train_q, val_q


@app.cell
def _(ALL_Q, FEATURES, LGBMRegressor, TARGET, np, train_q):
    # ---- Stage 1: one LightGBM per quantile, trained on log1p(demand) ----
    y_train_log = np.log1p(train_q[TARGET])

    q_models = {}
    for _q in ALL_Q:
        _m = LGBMRegressor(objective="quantile", alpha=_q, n_estimators=300, random_state=42, verbose=-1)
        _m.fit(train_q[FEATURES], y_train_log)
        q_models[_q] = _m

    print("trained quantile models:", list(q_models.keys()))
    return (q_models,)


@app.cell
def _(ALL_Q, FEATURES, TARGET, np, q_models, val_q):
    # ---- predict, convert back to real units, make sure the quantiles never cross ----
    _cols = [np.expm1(q_models[_q].predict(val_q[FEATURES])) for _q in ALL_Q]
    _raw = np.column_stack(_cols)                                  # one column per quantile, in ALL_Q order
    crossing_rate = (np.diff(_raw, axis=1) < 0).any(axis=1).mean() # share of rows where a higher quantile came out LOWER
    _fixed = np.sort(np.clip(_raw, 0, None), axis=1)               # sorting each row removes any crossing; clip keeps demand >= 0

    q_val = val_q[["time_idx", "Site", "Brand", "Range", "range_age_months"]].copy()
    for _c in ["Site", "Brand", "Range"]:
        q_val[_c] = q_val[_c].astype(str)
    for _i, _q in enumerate(ALL_Q):
        q_val[f"p{int(round(_q * 100))}"] = _fixed[:, _i]          # p10, p50, p80, p90, p95
    q_val["actual"] = val_q[TARGET].values
    q_val["lag_1"] = val_q["lag_1_gross_qty"].values

    print(f"rows where quantiles crossed before sorting: {crossing_rate:.1%}")
    q_val.head()
    return (q_val,)


@app.cell
def _(Z90, np):
    # ---- Stage 2: turn P10/P50/P90 into a two-piece curve on the log1p scale ----
    def fit_two_piece(p10, p50, p90):
        mu = np.log1p(p50)                                   # centre of the curve
        sig_l = (np.log1p(p50) - np.log1p(p10)) / Z90        # spread on the LEFT of the centre
        sig_r = (np.log1p(p90) - np.log1p(p50)) / Z90        # spread on the RIGHT of the centre
        return mu, sig_l, sig_r

    # ---- check against the teaching example: P10=120, P50=200, P90=360 ----
    _mu, _sl, _sr = fit_two_piece(np.array([120.0]), np.array([200.0]), np.array([360.0]))
    print(f"mu = {_mu[0]:.4f} (example: 5.3033) | sigma_left = {_sl[0]:.3f} (0.396) | sigma_right = {_sr[0]:.3f} (0.457)")
    assert abs(_mu[0] - 5.3033) < 1e-3 and abs(_sl[0] - 0.396) < 1e-3 and abs(_sr[0] - 0.457) < 1e-3
    print("example reproduced")
    return (fit_two_piece,)


@app.cell
def _(fit_two_piece, pd, q_val):
    # ---- apply the same fit to every validation row ----
    fit_val = pd.DataFrame(index=q_val.index)
    fit_val["mu"], fit_val["sig_l"], fit_val["sig_r"] = fit_two_piece(
        q_val["p10"].values, q_val["p50"].values, q_val["p90"].values
    )

    # zero-heavy check (the pitfall from Stage 2): how many rows does a bell curve struggle with?
    print(f"rows where the model's P50 is 0:      {(q_val['p50'] == 0).mean():.1%}")
    print(f"rows where the model's P10 is 0:      {(q_val['p10'] == 0).mean():.1%}")
    print(f"rows where actual demand really is 0: {(q_val['actual'] == 0).mean():.1%}")
    fit_val.describe().round(3)
    return (fit_val,)


@app.cell
def _(fit_val):
    fit_val.info()
    return


@app.cell
def _(fit_two_piece, np):
    # ---- Stage 3: Monte Carlo = draw many random outcomes from the fitted curve ----
    def samples_from_z(mu, sig_l, sig_r, z):
        sig = np.where(z < 0, sig_l[:,
            None], sig_r[:, None])        # left sigma for z<0, right sigma for z>=0
        return np.clip(np.expm1(mu[:, None] + sig * z), 0, None)     # back to real units, demand can't be negative

    def draw_samples(mu, sig_l, sig_r, n_draws, seed):
        rng = np.random.default_rng(seed)
        return samples_from_z(mu, sig_l, sig_r, rng.standard_normal((len(mu), n_draws)))

    # ---- check against the teaching example (200,000 draws for ONE row) ----
    _mu, _sl, _sr = fit_two_piece(np.array([120.0]), np.array([200.0]), np.array([360.0]))
    _s = draw_samples(_mu, _sl, _sr, 200_000, 7)[0]
    _got = {"P10": np.percentile(_s, 10), "P50": np.percentile(_s, 50), "P80": np.percentile(_s, 80),
            "P90": np.percentile(_s, 90), "P95": np.percentile(_s, 95), "P99": np.percentile(_s, 99),
            "mean": _s.mean(), "P(>300)": (_s > 300).mean()}
    _want = {"P10": 120, "P50": 200, "P80": 294, "P90": 360, "P95": 425, "P99": 581, "mean": 225, "P(>300)": 0.188}
    for _k in _got:
        print(f"{_k:8s} got {_got[_k]:8.3f} | example {_want[_k]}")
        assert abs(_got[_k] - _want[_k]) <= 0.03 * abs(_want[_k]), _k
    print("example reproduced (within 3% sampling noise)")
    return draw_samples, samples_from_z


@app.cell
def _(N_DRAWS, SEED, draw_samples, fit_val, np, pd, q_val):
    # ---- Monte Carlo for every validation row ----
    mc_val = draw_samples(
        fit_val["mu"].values,
        fit_val["sig_l"].values,
        fit_val["sig_r"].values,
        N_DRAWS,
        SEED
    )

    _pct = np.percentile(mc_val, [10, 50, 80, 90, 95], axis=1)
    mc_summary = pd.DataFrame(_pct.T, columns=["mc_p10", "mc_p50", "mc_p80", "mc_p90", "mc_p95"], index=q_val.index)
    mc_summary["mc_mean"] = mc_val.mean(axis=1)

    print("samples array (rows x draws):", mc_val.shape)
    mc_summary.describe().round(1)
    return mc_summary, mc_val


@app.cell
def _(mc_summary):
    mc_summary.info()
    return


@app.cell
def _():
    return


@app.cell
def _(mc_summary, pd, q_val):
    # ---- does the fitted curve agree with quantile models trained directly? ----
    # P10/P50/P90 were used to fit, so they must match. P80/P95 were NOT used: any gap is the curve-shape assumption.
    shape_test = pd.DataFrame({
        "trained_total": [q_val[_c].sum() for _c in ["p10", "p50", "p80", "p90", "p95"]],
        "monte_carlo_total": [mc_summary[_c].sum() for _c in ["mc_p10", "mc_p50", "mc_p80", "mc_p90", "mc_p95"]],
    }, index=["P10 (fit)", "P50 (fit)", "P80 (test)", "P90 (fit)", "P95 (test)"])
    shape_test["gap_%"] = 100 * (shape_test["monte_carlo_total"] / shape_test["trained_total"] - 1)
    shape_test.round(1)
    return


@app.cell
def _(fit_two_piece, np, samples_from_z):
    # ---- Stage 4 example: quantiles do not add. Two identical series, each P90 = 360 ----
    _mu, _sl, _sr = fit_two_piece(np.array([120.0]), np.array([200.0]), np.array([360.0]))
    _rng = np.random.default_rng(3)
    _z1 = _rng.standard_normal((1, 200_000))
    _z2_ind = _rng.standard_normal((1, 200_000))
    _z2_cor = 0.8 * _z1 + np.sqrt(1 - 0.8 ** 2) * _rng.standard_normal((1, 200_000))

    _a = samples_from_z(_mu, _sl, _sr, _z1)[0]
    _b_ind = samples_from_z(_mu, _sl, _sr, _z2_ind)[0]
    _b_cor = samples_from_z(_mu, _sl, _sr, _z2_cor)[0]

    print(f"single series P90:               {np.percentile(_a, 90):.0f}   (example 360)")
    print(f"sum of the two P90s:             {2 * np.percentile(_a, 90):.0f}   (example 720)")
    print(f"P90 of the SUM, independent:     {np.percentile(_a + _b_ind, 90):.0f}   (example 645)")
    print(f"P90 of the SUM, correlation 0.8: {np.percentile(_a + _b_cor, 90):.0f}   (example 705)")
    print(f"P90 of the SUM, correlation 1.0: {np.percentile(_a + _a, 90):.0f}   (example 720)")
    return


@app.cell
def _(draw_samples, fit_two_piece, np):
    # ---- evaluation toolbox: every metric from the teaching, one small function each ----
    def m_mae(y, p):  return np.mean(np.abs(p - y))                      # typical miss, in units
    def m_rmse(y, p): return np.sqrt(np.mean((p - y) ** 2))              # punishes big misses harder
    def m_wape(y, p): return np.abs(p - y).sum() / np.abs(y).sum()       # total miss / total demand
    def m_bias(y, p): return np.mean(p - y)                              # negative = under-forecasting

    def m_pinball(y, q, tau):                                            # asymmetric penalty for quantile q at level tau
        d = y - q
        return np.mean(np.maximum(tau * d, (tau - 1) * d))

    def m_coverage(y, lo, hi): return np.mean((y >= lo) & (y <= hi))     # share of actuals inside [lo, hi]
    def m_width(lo, hi):       return np.mean(hi - lo)                   # sharpness: narrower is better IF coverage holds

    def m_interval_score(y, lo, hi, alpha=0.2):                          # width + 10x the distance outside (alpha = 0.2)
        return np.mean((hi - lo) + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0))

    def m_crps(y, samples):                                              # per-row CRPS = E|X-y| - 0.5*E|X-X'|
        n = samples.shape[1]
        s = np.sort(samples, axis=1)
        term1 = np.mean(np.abs(s - y[:, None]), axis=1)
        i = np.arange(1, n + 1)
        term2 = (2.0 / n ** 2) * (s * (2 * i - n - 1)).sum(axis=1)       # E|X-X'| computed from sorted draws
        return term1 - 0.5 * term2

    def m_pit(y, samples, seed=0):                                       # where does each actual land inside its own forecast?
        rng = np.random.default_rng(seed)
        lower = (samples < y[:, None]).mean(axis=1)
        upper = (samples <= y[:, None]).mean(axis=1)
        return lower + rng.random(len(y)) * (upper - lower)              # random spread inside ties: fair when many actuals are 0

    # ---- check against the 8-row teaching example ----
    _y = np.array([40, 120, 15, 300, 80, 0, 700, 200], float)
    _p10 = np.array([25, 90, 5, 150, 60, 0, 250, 140], float)
    _p50 = np.array([45, 130, 20, 220, 95, 8, 380, 210], float)
    _p90 = np.array([80, 190, 50, 350, 150, 40, 560, 330], float)
    _checks = [
        ("MAE (P50)",        m_mae(_y, _p50),                    56.625),
        ("RMSE (P50)",       m_rmse(_y, _p50),                   116.9),
        ("WAPE (P50)",       m_wape(_y, _p50),                   0.311),
        ("Bias (P50)",       m_bias(_y, _p50),                   -43.375),
        ("pinball 0.1",      m_pinball(_y, _p10, 0.1),           9.19),
        ("pinball 0.5",      m_pinball(_y, _p50, 0.5),           28.31),
        ("pinball 0.9",      m_pinball(_y, _p90, 0.9),           21.19),
        ("coverage",         m_coverage(_y, _p10, _p90),         0.875),
        ("mean width",       m_width(_p10, _p90),                128.75),
        ("interval score",   m_interval_score(_y, _p10, _p90),   303.75),
    ]
    for _name, _got, _want in _checks:
        print(f"{_name:15s} got {_got:9.3f} | example {_want}")
        assert abs(_got - _want) < 0.01 * max(1, abs(_want)), _name

    # CRPS + PIT example: actual = 120 against the Stage-2 curve (P10=120, P50=200, P90=360)
    _mu2, _sl2, _sr2 = fit_two_piece(np.array([120.0]), np.array([200.0]), np.array([360.0]))
    _s2 = draw_samples(_mu2, _sl2, _sr2, 20_000, 11)
    _crps = m_crps(np.array([120.0]), _s2)[0]
    _pit = m_pit(np.array([120.0]), _s2)[0]
    print(f"{'CRPS (y=120)':15s} got {_crps:9.3f} | example 54.2")
    print(f"{'PIT  (y=120)':15s} got {_pit:9.3f} | example 0.099")
    assert abs(_crps - 54.2) < 1.5 and abs(_pit - 0.099) < 0.02
    print("all teaching examples reproduced")
    return (
        m_bias,
        m_coverage,
        m_crps,
        m_interval_score,
        m_mae,
        m_pinball,
        m_pit,
        m_rmse,
        m_wape,
        m_width,
    )


@app.cell
def _(m_bias, m_mae, m_rmse, m_wape, pd, q_val):
    # ---- Layer A: point accuracy of the median forecast (P50) ----
    _y = q_val["actual"].values
    _has_lag = q_val["lag_1"].notna().values            # the naive baseline needs last month's value

    _sets = {
        "P50 (all rows)":                (_y,            q_val["p50"].values),
        "P50 (rows with lag_1)":         (_y[_has_lag],  q_val["p50"].values[_has_lag]),
        "Naive lag_1 (rows with lag_1)": (_y[_has_lag],  q_val["lag_1"].values[_has_lag]),
    }
    table_a = pd.DataFrame({
        _name: {"MAE": m_mae(_yy, _pp), "RMSE": m_rmse(_yy, _pp), "WAPE": m_wape(_yy, _pp),
                "Bias": m_bias(_yy, _pp), "n_rows": len(_yy)}
        for _name, (_yy, _pp) in _sets.items()
    }).T
    table_a.round(3)
    return


@app.cell
def _(
    ALL_Q,
    FIT_Q,
    m_coverage,
    m_crps,
    m_interval_score,
    m_pinball,
    m_width,
    mc_val,
    np,
    pd,
    q_val,
):
    # ---- Layer B: is the uncertainty honest? ----
    _y = q_val["actual"].values
    _col = {tau: f"p{int(round(tau * 100))}" for tau in ALL_Q}

    pinball_by_tau = pd.Series({tau: m_pinball(_y, q_val[_col[tau]].values, tau) for tau in ALL_Q}, name="pinball")

    summary_b = pd.Series({
        "mean pinball (0.1, 0.5, 0.9)": np.mean([pinball_by_tau[t] for t in FIT_Q]),
        "coverage of [P10, P90]  (target 0.80)": m_coverage(_y, q_val["p10"].values, q_val["p90"].values),
        "mean width of [P10, P90]": m_width(q_val["p10"].values, q_val["p90"].values),
        "interval score (alpha 0.2)": m_interval_score(_y, q_val["p10"].values, q_val["p90"].values),
        "CRPS (Monte Carlo samples)": m_crps(_y, mc_val).mean(),
    })

    # reliability: what share of actuals fall below each quantile?  (strict "<" and inclusive "<=")
    # with many true zeros, "equal to the quantile" is common, so calibrated means: nominal sits BETWEEN the two columns
    reliability = pd.DataFrame({
        "nominal": ALL_Q,
        "observed_below_strict": [np.mean(_y < q_val[_col[t]].values) for t in ALL_Q],
        "observed_below_inclusive": [np.mean(_y <= q_val[_col[t]].values) for t in ALL_Q],
    }, index=[_col[t] for t in ALL_Q])

    print(pinball_by_tau.round(2).to_string())
    print()
    print(summary_b.round(3).to_string())
    reliability.round(3)
    return (reliability,)


@app.cell
def _(m_pit, mc_val, np, pd, q_val):
    # ---- PIT: where does each actual land inside its own Monte Carlo forecast? ----
    pit_val = m_pit(q_val["actual"].values, mc_val)

    _counts, _edges = np.histogram(pit_val, bins=10, range=(0, 1))
    pit_table = pd.DataFrame({
        "bin": [f"{_edges[_i]:.1f}-{_edges[_i + 1]:.1f}" for _i in range(10)],
        "share_of_rows": _counts / len(pit_val),
        "calibrated_would_be": 0.10,
    })
    pit_table.round(3)
    return (pit_val,)


@app.cell
def _(m_coverage, mc_val, np, pd, q_val):
    # ---- Stage 4 in practice: totals per Site x month, two ways of building the P10-P90 band ----
    _groups = q_val.groupby(["Site", "time_idx"]).indices        # row positions of each Site x month
    _rows = []
    for _key, _idx in _groups.items():
        _tot_samples = mc_val[_idx].sum(axis=0)                  # add the DRAWS first, then read quantiles (independent rows)
        _rows.append({
            "actual_total": q_val["actual"].values[_idx].sum(),
            "mc_p10": np.percentile(_tot_samples, 10), "mc_p90": np.percentile(_tot_samples, 90),
            "sumq_p10": q_val["p10"].values[_idx].sum(), "sumq_p90": q_val["p90"].values[_idx].sum(),
            "n_series": len(_idx),
        })
    agg_val = pd.DataFrame(_rows)

    agg_summary = pd.Series({
        "Site-month totals": len(agg_val),
        "coverage, band from summed Monte Carlo draws (target 0.80)": m_coverage(agg_val["actual_total"], agg_val["mc_p10"], agg_val["mc_p90"]),
        "coverage, band from summed quantiles (target 0.80)": m_coverage(agg_val["actual_total"], agg_val["sumq_p10"], agg_val["sumq_p90"]),
    })
    agg_summary.round(3)
    return


@app.cell
def _(q_val):
    # ---- Layer D prep: one row per validation row with its error and whether it was covered ----
    seg_q = q_val.copy()
    seg_q["abs_err"] = (seg_q["p50"] - seg_q["actual"]).abs()
    seg_q["inside"] = (seg_q["actual"] >= seg_q["p10"]) & (seg_q["actual"] <= seg_q["p90"])
    seg_q["width"] = seg_q["p90"] - seg_q["p10"]
    seg_q.head()
    return (seg_q,)


@app.cell
def _(pd, seg_q):
    # ---- Layer D: by demand size (same 10 deciles of NON-ZERO rows as the baseline comparison) ----
    _nz = seg_q[seg_q["actual"] > 0].copy()
    _nz["decile"] = pd.qcut(_nz["actual"].rank(method="first"), 10, labels=False)

    table_decile = _nz.groupby("decile").agg(
        actual_min=("actual", "min"), actual_max=("actual", "max"), actual_sum=("actual", "sum"),
        abs_err_sum=("abs_err", "sum"), MAE=("abs_err", "mean"),
        coverage=("inside", "mean"), mean_width=("width", "mean"), n_rows=("actual", "size"),
    )
    table_decile["WAPE"] = table_decile["abs_err_sum"] / table_decile["actual_sum"]
    table_decile["pct_of_volume"] = 100 * table_decile["actual_sum"] / table_decile["actual_sum"].sum()
    table_decile.drop(columns="abs_err_sum").round(3)
    return (table_decile,)


@app.cell
def _(np, pd, seg_q):
    # ---- Layer D: other cuts - zero vs non-zero rows, Range age, month ----
    _zero = seg_q["actual"] == 0
    table_zero = pd.DataFrame({
        "actual == 0": {"n_rows": _zero.sum(), "coverage": seg_q.loc[_zero, "inside"].mean(),
                        "share where P10 > 0 (forecast expects sales, there were none)": (seg_q.loc[_zero, "p10"] > 0).mean(),
                        "mean P50": seg_q.loc[_zero, "p50"].mean()},
        "actual > 0":  {"n_rows": (~_zero).sum(), "coverage": seg_q.loc[~_zero, "inside"].mean(),
                        "share where P10 > 0 (forecast expects sales, there were none)": np.nan,
                        "mean P50": seg_q.loc[~_zero, "p50"].mean()},
    }).T

    _age = pd.cut(seg_q["range_age_months"], [-1, 2, 11, 10_000], labels=["0-2 (new)", "3-11", "12+ (established)"])
    table_age = seg_q.groupby(_age, observed=True).agg(
        n_rows=("actual", "size"), actual_sum=("actual", "sum"), abs_err_sum=("abs_err", "sum"),
        coverage=("inside", "mean"), mean_width=("width", "mean"),
    )
    table_age["WAPE"] = table_age["abs_err_sum"] / table_age["actual_sum"]

    table_month = seg_q.groupby("time_idx").agg(
        n_rows=("actual", "size"), actual_sum=("actual", "sum"), abs_err_sum=("abs_err", "sum"), coverage=("inside", "mean"),
    )
    table_month["WAPE"] = table_month["abs_err_sum"] / table_month["actual_sum"]

    print(table_zero.round(3).to_string()); print()
    print(table_age.drop(columns="abs_err_sum").round(3).to_string()); print()
    print(table_month.drop(columns="abs_err_sum").round(3).to_string())
    return (table_age,)


@app.cell
def _(
    CAT_COLS,
    FEATURES,
    FIT_Q,
    LGBMRegressor,
    TARGET,
    VAL_MONTHS,
    data_q,
    m_bias,
    m_coverage,
    m_mae,
    m_pinball,
    m_wape,
    np,
    pd,
):
    # ---- Layer D: rolling-origin backtest - retrain at several cut-off months, always test on the NEXT 6 months ----
    def run_window(cutoff):
        tr = data_q[data_q["time_idx"] <= cutoff].copy()
        va = data_q[(data_q["time_idx"] > cutoff) & (data_q["time_idx"] <= cutoff + VAL_MONTHS)].copy()
        for c in CAT_COLS:
            tr[c] = tr[c].astype(str).astype("category")
            va[c] = pd.Categorical(va[c].astype(str), categories=tr[c].cat.categories)
        ylog = np.log1p(tr[TARGET])
        out = {}
        for q in FIT_Q:
            m = LGBMRegressor(objective="quantile", alpha=q, n_estimators=300, random_state=42, verbose=-1)
            m.fit(tr[FEATURES], ylog)
            out[q] = np.expm1(m.predict(va[FEATURES]))
        pred = np.sort(np.clip(np.column_stack([out[q] for q in FIT_Q]), 0, None), axis=1)
        y = va[TARGET].values
        return {
            "train_months": int(cutoff + 1), "val_rows": len(va),
            "WAPE (P50)": m_wape(y, pred[:, 1]), "MAE (P50)": m_mae(y, pred[:, 1]), "Bias (P50)": m_bias(y, pred[:, 1]),
            "coverage [P10,P90]": m_coverage(y, pred[:, 0], pred[:, 2]),
            "mean pinball": np.mean([m_pinball(y, pred[:, i], FIT_Q[i]) for i in range(3)]),
        }

    _last = data_q["time_idx"].max()
    _cutoffs = [_last - 24, _last - 18, _last - 12, _last - 6]
    table_windows = pd.DataFrame({f"cutoff {_c}": run_window(_c) for _c in _cutoffs}).T
    table_windows.round(3)
    return


@app.cell
def _(mc_val, np, pd, plt, q_val):
    # ---- Plot 1: fan chart - monthly TOTAL actual vs forecast, with the P10-P90 band ----
    _months = sorted(q_val["time_idx"].unique())
    _rows = []
    for _t in _months:
        _idx = np.where(q_val["time_idx"].values == _t)[0]
        _tot = mc_val[_idx].sum(axis=0)                                   # add the draws, then read the band
        _rows.append({"time_idx": _t,
                      "actual": q_val["actual"].values[_idx].sum(),
                      "p50_sum": q_val["p50"].values[_idx].sum(),
                      "mc_p10": np.percentile(_tot, 10), "mc_p90": np.percentile(_tot, 90),
                      "sumq_p10": q_val["p10"].values[_idx].sum(), "sumq_p90": q_val["p90"].values[_idx].sum()})
    fan_df = pd.DataFrame(_rows)

    fig_fan, ax_fan = plt.subplots(figsize=(8, 4.5))
    ax_fan.fill_between(fan_df["time_idx"], fan_df["mc_p10"], fan_df["mc_p90"], alpha=0.25, label="P10-P90 (summed Monte Carlo draws)")
    ax_fan.plot(fan_df["time_idx"], fan_df["sumq_p10"], linestyle=":", color="gray", label="sum of row P10 / P90")
    ax_fan.plot(fan_df["time_idx"], fan_df["sumq_p90"], linestyle=":", color="gray")
    ax_fan.plot(fan_df["time_idx"], fan_df["p50_sum"], marker="o", label="Forecast (sum of P50)")
    ax_fan.plot(fan_df["time_idx"], fan_df["actual"], marker="o", color="black", label="Actual")
    ax_fan.set_xlabel("time_idx"); ax_fan.set_ylabel("Total gross_qty"); ax_fan.set_title("Fan chart: actual vs forecast band")
    ax_fan.legend()
    fig_fan
    return


@app.cell
def _(plt, q_val):
    # ---- Plot 2: predicted vs actual, every row ----
    fig_sc, ax_sc = plt.subplots(figsize=(6, 6))
    ax_sc.scatter(q_val["actual"], q_val["p50"], alpha=0.3, s=8)
    _top = max(q_val["actual"].max(), q_val["p50"].max())
    ax_sc.plot([0, _top], [0, _top], color="red", linestyle="--", label="perfect (y = x)")
    ax_sc.set_xscale("symlog"); ax_sc.set_yscale("symlog")
    ax_sc.set_xlabel("Actual (symlog)"); ax_sc.set_ylabel("Predicted P50 (symlog)"); ax_sc.set_title("Predicted vs actual")
    ax_sc.legend()
    fig_sc
    return


@app.cell
def _(plt, q_val):
    # ---- Plot 3: residual (predicted - actual) vs actual ----
    residual_q = q_val["p50"].values - q_val["actual"].values
    fig_res, ax_res = plt.subplots(figsize=(8, 4.5))
    ax_res.scatter(q_val["actual"], residual_q, alpha=0.3, s=8)
    ax_res.axhline(0, color="red", linestyle="--", label="zero error")
    ax_res.set_xscale("symlog"); ax_res.set_yscale("symlog")
    ax_res.set_xlabel("Actual (symlog)"); ax_res.set_ylabel("Predicted - actual (symlog)"); ax_res.set_title("Residuals vs actual demand")
    ax_res.legend()
    fig_res
    return


@app.cell
def _(plt, table_decile):
    # ---- Plot 4: MAE by demand decile, with each decile's share of total volume ----
    fig_dec, ax_dec = plt.subplots(figsize=(8, 4.5))
    ax_dec.bar(table_decile.index, table_decile["MAE"])
    ax_dec.set_xlabel("Demand decile (0 = smallest non-zero actuals)"); ax_dec.set_ylabel("MAE (units)")
    ax_dec2 = ax_dec.twinx()
    ax_dec2.plot(table_decile.index, table_decile["pct_of_volume"], color="red", marker="o")
    ax_dec2.set_ylabel("% of total volume", color="red")
    ax_dec.set_title("MAE by decile (bars) and share of volume (red)")
    fig_dec
    return


@app.cell
def _(plt, reliability):
    # ---- Plot 5: reliability - does a "P90" really have 90% of actuals below it? ----
    fig_rel, ax_rel = plt.subplots(figsize=(5.5, 5.5))
    ax_rel.plot([0, 1], [0, 1], color="red", linestyle="--", label="perfect")
    ax_rel.plot(reliability["nominal"], reliability["observed_below_strict"], marker="o", label="observed share below (<)")
    ax_rel.plot(reliability["nominal"], reliability["observed_below_inclusive"], marker="s", label="observed share below (<=)")
    ax_rel.set_xlabel("Nominal quantile level"); ax_rel.set_ylabel("Observed share of actuals below it"); ax_rel.set_title("Reliability")
    ax_rel.legend()
    fig_rel
    return


@app.cell
def _(np, pit_val, plt):
    # ---- Plot 6: PIT histogram ----
    fig_pit, ax_pit = plt.subplots(figsize=(6.5, 4))
    ax_pit.hist(pit_val, bins=10, range=(0, 1), weights=np.ones(len(pit_val)) / len(pit_val), edgecolor="white")
    ax_pit.axhline(0.10, color="red", linestyle="--", label="flat = calibrated")
    ax_pit.set_xlabel("PIT (position of the actual inside its forecast)"); ax_pit.set_ylabel("Share of rows"); ax_pit.set_title("PIT histogram")
    ax_pit.legend()
    fig_pit
    return


@app.cell
def _(plt, table_age, table_decile):
    # ---- Plot 7: coverage by segment - where does the P10-P90 band fail? ----
    fig_cov, (ax_cov1, ax_cov2) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    ax_cov1.bar(table_decile.index, table_decile["coverage"])
    ax_cov1.axhline(0.8, color="red", linestyle="--", label="target 80%")
    ax_cov1.set_xlabel("Demand decile"); ax_cov1.set_ylabel("Coverage"); ax_cov1.set_title("By demand size"); ax_cov1.legend()
    ax_cov2.bar([str(_a) for _a in table_age.index], table_age["coverage"])
    ax_cov2.axhline(0.8, color="red", linestyle="--")
    ax_cov2.set_xlabel("Range age (months)"); ax_cov2.set_title("By Range age")
    fig_cov.tight_layout()
    fig_cov
    return


@app.cell
def _(plt, seg_q):
    # ---- Plot 8: does the model know when it is unsure? (band width vs actual error) ----
    _wd = seg_q[seg_q["width"] > 0]
    _rank_corr = _wd["width"].rank().corr(_wd["abs_err"].rank())            # Spearman = Pearson on ranks
    fig_wid, ax_wid = plt.subplots(figsize=(7, 5))
    ax_wid.scatter(_wd["width"], _wd["abs_err"], alpha=0.3, s=8)
    ax_wid.set_xscale("symlog"); ax_wid.set_yscale("symlog")
    ax_wid.set_xlabel("Band width P90 - P10 (symlog)"); ax_wid.set_ylabel("Absolute error of P50 (symlog)")
    ax_wid.set_title(f"Width vs error (rank correlation {_rank_corr:.2f})")
    fig_wid
    return


@app.cell
def _():

    import torch
    import torch.nn.functional as F
    from torch import nn

    return F, nn, torch


@app.cell
def _(torch):
    def make_tensors(data):
        X = torch.tensor(data["X"], dtype=torch.float32)
        M = torch.tensor(data["M"], dtype=torch.float32)
        C = torch.tensor(data["C"], dtype=torch.float32)
        return X, M, C

    return (make_tensors,)


@app.cell
def _(torch):
    def pinball(pred, y, tau):
        d = y.unsqueeze(1) - pred
        return torch.maximum(tau * d, (tau - 1) * d).mean()

    return (pinball,)


@app.cell
def _(F, nn, torch):
    class Block(nn.Module):
        def __init__(self, L, n_cov, n_levels, pool, hidden=64):
            super().__init__()
            assert L % pool == 0, "window length must be divisible by the pool size"
            self.pool = pool
            self.mlp = nn.Sequential(nn.Linear(2 * (L // pool) + n_cov, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU())
            self.back = nn.Linear(hidden, L)
            self.fore = nn.Linear(hidden, n_levels)

        def forward(self, x, m, c):
            px = F.avg_pool1d(x.unsqueeze(1), self.pool).squeeze(1)
            pm = F.avg_pool1d(m.unsqueeze(1), self.pool).squeeze(1)
            h = self.mlp(torch.cat([px, pm, c], dim=1))
            return self.back(h), self.fore(h)

    return (Block,)


@app.cell
def _(Block, nn):
    class NHiTS(nn.Module):
        def __init__(self, L, n_cov, n_levels, pools=(3, 1), hidden=64):
            super().__init__()
            self.blocks = nn.ModuleList([Block(L, n_cov, n_levels, p, hidden) for p in pools])

        def forward(self, x, m, c):
            forecast = 0
            for block in self.blocks:
                back, f = block(x, m, c)
                x = (x - back) * m
                forecast = forecast + f
            return forecast

    return (NHiTS,)


@app.cell
def _(NHiTS, make_tensors, pinball, torch):
    def train_one(data, levels, seed, epochs=200, lr=1e-3, patience=20, batch=256, pools=(3, 1), hidden=64):
        torch.manual_seed(seed)
        X, M, C = make_tensors(data)
        y = torch.tensor(data["y"], dtype=torch.float32)
        tau = torch.tensor(levels, dtype=torch.float32)
        is_train = torch.tensor(data["t"] <= data["t"].max() - 3)
        tr_idx, va_idx = torch.where(is_train)[0], torch.where(~is_train)[0]
        model = NHiTS(X.shape[1], C.shape[1], len(levels), pools, hidden)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        best, best_state, wait = float("inf"), None, 0
        for epoch in range(epochs):
            model.train()
            order = tr_idx[torch.randperm(len(tr_idx))]
            for i in range(0, len(order), batch):
                b = order[i:i + batch]
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
                if wait >= patience:
                    break
        model.load_state_dict(best_state)
        return model.eval()

    return (train_one,)


@app.cell
def _(make_tensors, np, torch, train_one):
    def train_nhits(data, levels, n_seeds=3):
        return [train_one(data, levels, seed) for seed in range(n_seeds)]

    def predict_nhits(models, data):
        X, M, C = make_tensors(data)
        with torch.no_grad():
            return np.mean([m(X, M, C).numpy() for m in models], axis=0)

    return


@app.cell
def _():
    from models.windows import WindowBuilder

    return (WindowBuilder,)


@app.cell
def _(WindowBuilder, np, panel, train_w):

    needed = ["Site", "Brand", "Range", "time_idx", "gross_qty", "month_sin", "month_cos", "range_age_months"]
    assert all(c in panel.columns for c in needed), [c for c in needed if c not in panel.columns]

    builder = WindowBuilder(window=6)
    d_train = builder.arrays(train_w, train_w)             # one window per training row
    keep = d_train["ok"]                                    # drop rows with no earlier month (age 0)
    data_train = {k: v[keep] for k, v in d_train.items() if k != "ok"}

    print("shapes:", {k: v.shape for k, v in data_train.items()})
    print("rows kept:", int(keep.sum()), "of", len(keep), "| NaN anywhere:", any(bool(np.isnan(v).any()) for v in data_train.values()))
    print("padded windows: %.1f%%" % (100 * (data_train["M"].sum(axis=1) < 6).mean()),
          "| y range:", round(data_train["y"].min(), 2), "to", round(data_train["y"].max(), 2),
          "| months", data_train["t"].min(), "to", data_train["t"].max())
    return


@app.cell
def _(Path, pd):
    LEVELS = [0.1, 0.5, 0.8, 0.9, 0.95]

    panel = pd.read_pickle(Path("data/processed/training_data.pkl")).dropna(subset=["gross_qty"])
    cutoff_dev = int(panel["time_idx"].max()) - 6
    train_w = panel[panel["time_idx"] <= cutoff_dev]      # months up to 35
    val_dev = panel[panel["time_idx"] > cutoff_dev]       # months 36-41
    print(panel.shape, "| train rows:", len(train_w), "| validation rows:", len(val_dev))
    return panel, train_w


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
