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
    import pandas as pd

    project_root = Path(__file__).resolve().parent.parent
    monthly_demand = pd.read_pickle(project_root / "data" / "processed" / "monthly_demand.pkl")
    return monthly_demand, pd


@app.cell
def _(monthly_demand):
    df=monthly_demand
    return (df,)


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df):
    val_months=6
    cutoff = df["time_idx"].max() - val_months
    train_split =df[df["time_idx"] <= cutoff]
    val_split   =df[df["time_idx"] > cutoff]

    features = ["year", "month_sin", "month_cos", "time_idx", "range_age_months",
                "lag_1_gross_qty", "rolling_mean_3", "rolling_mean_6", "lag_12_gross_qty",
                "return_rate_prev_month", "Site", "Brand", "Range"]
    target = "gross_qty"

    for d_f in [train_split, val_split]:
        for col in ["Site", "Brand", "Range"]:
            d_f[col] = d_f[col].astype("category")   # tells LightGBM to treat these as categories, not numbers
    return features, target, train_split, val_split


@app.cell
def _(train_split):
    train_split.info()
    return


@app.cell
def _(val_split):
    val_split.info()
    return


@app.cell
def _():
    from lightgbm import LGBMRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return LGBMRegressor, mean_absolute_error, mean_squared_error, r2_score


@app.cell
def _(LGBMRegressor, features, target, train_split):
    model = LGBMRegressor(
        objective="regression",

        # Number of boosting rounds
        n_estimators=550,
        learning_rate=0.06,

        # Tree complexity
        num_leaves=15,
        max_depth=20,
        min_child_samples=40,

        # Stronger regularization
        reg_alpha=5.0,
        reg_lambda=10.0,
        min_split_gain=1.0,

        # Sampling to reduce overfitting
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,

        random_state=42,
        verbosity=-1
    )
    model.fit(train_split[features], train_split[target])
    return (model,)


@app.cell
def _(features, model, val_split):
    pred_model = model.predict(val_split[features])
    pred_naive = val_split["lag_1_gross_qty"]
    return pred_model, pred_naive


@app.cell
def _(
    mean_absolute_error,
    mean_squared_error,
    pd,
    pred_model,
    pred_naive,
    r2_score,
    target,
    val_split,
):

    import numpy as np


    valid_mask = pred_naive.notna()           # same mask as last message - 56 rows naive can't attempt
    y_true        = val_split.loc[valid_mask, target]
    pred_model_f  = pred_model[valid_mask.values]
    pred_naive_f  = pred_naive[valid_mask]

    def wape(y_true, y_pred):                 # Weighted Absolute Percentage Error
        return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()

    def bias(y_true, y_pred):                 # mean signed error - over- vs under-predicting
        return (y_pred - y_true).mean()

    results = {}
    for name, pred in [("Naive (lag_1)", pred_naive_f), ("LightGBM", pred_model_f)]:
        results[name] = {
            "MAE":  mean_absolute_error(y_true, pred),
            "RMSE": mean_squared_error(y_true, pred) ** 0.5,
            "WAPE": wape(y_true, pred),
            "R2":   r2_score(y_true, pred),
            "Bias": bias(y_true, pred),
        }

    pd.DataFrame(results).T
    return bias, np, wape


@app.cell
def _(pred_model, pred_naive, val_split):
    import matplotlib.pyplot as plt

    val_split2 = val_split.copy()
    val_split2["pred_model"] = pred_model
    val_split2["pred_naive"] = pred_naive

    # aggregate across every Site×Brand×Range, one total per month - a "trend" needs that,
    # per-row would just be thousands of overlapping dots with no shape to read
    trend = val_split2.groupby("time_idx").agg(
        actual=("gross_qty", "sum"),
        predicted=("pred_model", "sum"),
        naive=("pred_naive", "sum"),
    ).reset_index()

    fig, ax = plt.subplots()
    ax.plot(trend["time_idx"], trend["actual"], marker="o", label="Actual")
    ax.plot(trend["time_idx"], trend["predicted"], marker="o", label="LightGBM")
    ax.plot(trend["time_idx"], trend["naive"], marker="o", linestyle="--", label="Naive (lag_1)")
    ax.legend()
    ax.set_xlabel("time_idx")
    ax.set_ylabel("Total gross_qty")
    ax.set_title("Actual vs predicted - validation months")
    fig
    return plt, val_split2


@app.cell
def _(plt, val_split2):
    fig2, ax2 = plt.subplots()
    ax2.scatter(val_split2["gross_qty"], val_split2["pred_model"], alpha=0.3, s=10)
    lims = [0, val_split2["gross_qty"].max()]
    ax2.plot(lims, lims, color="red", linestyle="--", label="Perfect prediction (y = x)")
    ax2.set_xlabel("Actual gross_qty")
    ax2.set_ylabel("Predicted gross_qty")
    ax2.legend()
    fig2
    return


@app.cell
def _(plt, pred_model, val_split):
    # Cell 1 — the error itself
    residual3 = pred_model - val_split["gross_qty"].values
    # Cell 2 — the plot
    fig3, ax3 = plt.subplots()
    ax3.scatter(val_split["gross_qty"], residual3, alpha=0.3, s=5)
    # Cell 3 — scale, labels, and reference line
    ax3.axhline(0, color="red", linestyle="--", label="Zero error")
    ax3.set_xscale("symlog")
    ax3.set_xlabel("Actual gross_qty (symlog scale)")
    ax3.set_ylabel("Prediction error (Predicted − Actual)")
    ax3.set_title("Residuals vs actual demand")
    ax3.legend()
    fig3
    return


@app.cell
def _(pd, val_split):
    val_split["decile"] = pd.qcut(
        val_split["gross_qty"].rank(method="first"),
        10,
        labels=False
    )
    return


@app.cell
def _(pd, pred_model, val_split):
    val_split["pred_model"] = pred_model   # attach here - groupby needs it as a column, not a loose array
    val_split["decile"] = pd.qcut(val_split["gross_qty"].rank(method="first"), 10, labels=False)

    val_split.groupby("decile").apply(
        lambda g: pd.Series({
            "WAPE": (g["gross_qty"] - g["pred_model"]).abs().sum() / g["gross_qty"].abs().sum(),
            "n_rows": len(g)
        })
    )
    return


@app.cell
def _(val_split):
    print(val_split.columns.tolist())
    return


@app.cell
def _(val_split):
    (val_split["gross_qty"] == 0).mean()
    return


@app.cell
def _(pd, val_split):
    zero_mask = val_split["gross_qty"] == 0
    print(f"Zero-actual rows: {zero_mask.mean():.1%} of val_split")

    # For true-zero rows: WAPE is undefined, use MAE instead - how far off is the model when the truth is 0?
    mae_on_zeros = val_split.loc[zero_mask, "pred_model"].abs().mean()
    print(f"MAE on zero-actual rows (ideally close to 0): {mae_on_zeros:.2f}")

    # For everything else: NOW decile it - every bucket is guaranteed nonzero, no more inf
    nonzero = val_split[~zero_mask].copy()
    nonzero["decile"] = pd.qcut(nonzero["gross_qty"].rank(method="first"), 10, labels=False)
    nonzero.groupby("decile").apply(lambda g: pd.Series({
        "WAPE": (g["gross_qty"] - g["pred_model"]).abs().sum() / g["gross_qty"].abs().sum(),
        "n_rows": len(g),
        "actual_range": f"{g['gross_qty'].min():.0f}-{g['gross_qty'].max():.0f}"
    }))
    return nonzero, zero_mask


@app.cell
def _(nonzero, pd):
    decile_check = nonzero.groupby("decile").apply(lambda g: pd.Series({
        "actual_sum": g["gross_qty"].sum(),
        "MAE": (g["gross_qty"] - g["pred_model"]).abs().mean(),
        "WAPE": (g["gross_qty"] - g["pred_model"]).abs().sum() / g["gross_qty"].abs().sum(),
        "n_rows": len(g),
    }))
    decile_check["pct_of_total_volume"] = decile_check["actual_sum"] / decile_check["actual_sum"].sum() * 100
    decile_check
    return


@app.cell
def _(LGBMRegressor, features, np, target, train_split, val_split):

    train_split["log_target"] = np.log1p(train_split[target])

    model_log = LGBMRegressor(n_estimators=300, random_state=42)
    model_log.fit(train_split[features], train_split["log_target"])

    pred_log = model_log.predict(val_split[features])
    pred_model_log = np.expm1(pred_log)   # back to real units
    return model_log, pred_model_log


@app.cell
def _(features, model_log, np, val_split):
    # Cell 1 — convert predictions back to real units
    pred_model_log2 = np.expm1(model_log.predict(val_split[features]))
    return (pred_model_log2,)


@app.cell
def _(
    bias,
    mean_absolute_error,
    mean_squared_error,
    pred_model_log,
    pred_model_log2,
    r2_score,
    target,
    val_split,
    wape,
):
    # Cell 2 — same 5 metrics as the baseline comparison, now for the log model


    results_log = {
        "MAE":  mean_absolute_error(val_split[target], pred_model_log2),
        "RMSE": mean_squared_error(val_split[target], pred_model_log2) ** 0.5,
        "WAPE": wape(val_split[target], pred_model_log),
        "R2":   r2_score(val_split[target], pred_model_log2),
        "Bias": bias(val_split[target], pred_model_log2),
    }
    print(results_log)
    return


@app.cell
def _(plt, pred_model_log2, val_split):
    # Cell 1 — the error itself
    residual4 = pred_model_log2 - val_split["gross_qty"].values
    # Cell 2 — the plot
    fig4, ax4 = plt.subplots()
    ax4.scatter(val_split["gross_qty"], residual4, alpha=0.3, s=5)
    # Cell 3 — scale, labels, and reference line
    ax4.axhline(0, color="red", linestyle="--", label="Zero error")
    ax4.set_xscale("symlog")
    ax4.set_xlabel("Actual gross_qty (symlog scale)")
    ax4.set_ylabel("Prediction error (Predicted − Actual)")
    ax4.set_title("Residuals vs actual demand")
    ax4.legend()
    fig4
    return


@app.cell
def _(LGBMRegressor, features, target, train_split, val_split):


    model_tweedie = LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.3,
        n_estimators=300,
        random_state=42
    )

    model_tweedie.fit(
        train_split[features],
        train_split[target]
    )

    predictions = model_tweedie.predict(
        val_split[features]
    )
    return (predictions,)


@app.cell
def _(predictions, val_split):
    val_split["predictions"]=predictions
    return


@app.cell
def _(
    bias,
    mean_absolute_error,
    mean_squared_error,
    predictions,
    r2_score,
    target,
    val_split,
    wape,
):
    # Cell 2 — same 5 metrics as the baseline comparison, now for the log model


    results_log2 = {
        "MAE":  mean_absolute_error(val_split[target], predictions),
        "RMSE": mean_squared_error(val_split[target], predictions) ** 0.5,
        "WAPE": wape(val_split[target], predictions),
        "R2":   r2_score(val_split[target],predictions),
        "Bias": bias(val_split[target],predictions),
    }
    print(results_log2)
    return


@app.cell
def _(pred_model_log2, val_split):
    val_split["pred_model_log2"] = pred_model_log2
    return


@app.cell
def _():
    # Cell 3 — the real test, all three models side by side on the same deciles




    return


@app.cell
def _():
    return


@app.cell
def _(pd, predictions, val_split, zero_mask):
    val_split["pred_model_tweedie"] = predictions
    nonzero_all = val_split[~zero_mask].copy()
    nonzero_all["decile"] = pd.qcut(nonzero_all["gross_qty"].rank(method="first"), 10, labels=False)

    decile_compare_all = nonzero_all.groupby("decile").apply(lambda g: pd.Series({
        "actual_sum": g["gross_qty"].sum(),
        "MAE_original":  (g["gross_qty"] - g["pred_model"]).abs().mean(),
        "MAE_log":       (g["gross_qty"] - g["pred_model_log2"]).abs().mean(),
        "MAE_tweedie":   (g["gross_qty"] - g["pred_model_tweedie"]).abs().mean(),
        "WAPE_original": (g["gross_qty"] - g["pred_model"]).abs().sum() / g["gross_qty"].abs().sum(),
        "WAPE_log":      (g["gross_qty"] - g["pred_model_log2"]).abs().sum() / g["gross_qty"].abs().sum(),
        "WAPE_tweedie":  (g["gross_qty"] - g["pred_model_tweedie"]).abs().sum() / g["gross_qty"].abs().sum(),
        "n_rows": len(g),
    }))
    decile_compare_all
    return


@app.cell
def _(LGBMRegressor, features, np, target, train_split):
    # Cell A: 3 quantile models on log1p(y)
    train_log_target = np.log1p(train_split[target])
    quantile_models_log = {}
    for q in [0.1, 0.5, 0.9]:
        m = LGBMRegressor(objective="quantile", alpha=q, n_estimators=300, random_state=42)
        m.fit(train_split[features], train_log_target)
        quantile_models_log[q] = m
    return (quantile_models_log,)


@app.cell
def _(features, np, pd, quantile_models_log, val_split):
    # Cell B: predict, convert back, sort so quantiles can't cross
    raw_q = np.column_stack([
        np.expm1(quantile_models_log[q].predict(val_split[features])) for q in [0.1, 0.5, 0.9]
    ])
    q_preds = pd.DataFrame(np.sort(np.clip(raw_q, 0, None), axis=1),
                           columns=["p10", "p50", "p90"], index=val_split.index)
    return (q_preds,)


@app.cell
def _(mean_absolute_error, q_preds, target, val_split):
    # Cell C: coverage and median accuracy
    inside = ((val_split[target] >= q_preds["p10"]) & (val_split[target] <= q_preds["p90"])).mean()
    print(f"Coverage of [P10, P90]: {inside:.1%}  (target about 80%)")
    print(f"P50 MAE: {mean_absolute_error(val_split[target], q_preds['p50']):.1f}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
