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

    files = sorted(Path("data/raw").glob("*.xlsx"))

    dfs = []
    for f in files:
        sheets = pd.read_excel(f, sheet_name=None)      # None = read every sheet, not just the first
        for sheet_name, df in sheets.items():
            df = df.copy()
            df["source_file"] = f.name
            df["source_sheet"] = sheet_name
            dfs.append(df)

    first_cols = set(dfs[0].columns) - {"source_file", "source_sheet"}
    for df in dfs[1:]:
        assert set(df.columns) - {"source_file", "source_sheet"} == first_cols, "Column mismatch between sheets!"

    raw = pd.concat(dfs, ignore_index=True)

    expected = sum(len(df) for df in dfs)
    assert len(raw) == expected
    print(f"Merged {len(files)} files, {len(dfs)} sheets total → {len(raw):,} rows")
    return Path, files, pd, raw


@app.cell
def _(raw):
    raw.isnull().sum()
    return


@app.cell
def _(raw):
    raw.size
    return


@app.cell
def _(raw):
    blank_like = ["", "nan", "NULL", "null", "N/A", "n/a", "-"]
    for col in raw.select_dtypes(include=["object", "str"]).columns:
        n = raw[col].astype(str).str.strip().isin(blank_like).sum()
        if n:
            print(f"{col}: {n} blank-looking values")
    return


@app.cell
def _(raw):
    raw["Qty"].describe()
    return


@app.cell
def _(raw):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.hist(raw["Qty"], bins=40)
    ax
    return fig, plt


@app.cell
def _(raw):
    raw[["Customer", "Site", "Part", "Category", "Brand", "Range", "Style"]].nunique()
    return


@app.cell
def _(raw):
    raw["Date"].min(), raw["Date"].max()
    return


@app.cell
def _(plt, raw):
    fig1, ax1 = plt.subplots()
    ax1.scatter(range(len(raw)), raw["Qty"], s=5, alpha=0.4)
    ax1
    return


@app.cell
def _(plt, raw):
    fig2, ax2 = plt.subplots()
    ax2.boxplot(raw["Qty"], vert=False)
    ax2
    return


@app.cell
def _(raw):
    q1, q3 = raw["Qty"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    print(f"bounds: below {lower} or above {upper}")

    outliers = raw[(raw["Qty"] < lower) | (raw["Qty"] > upper)]
    len(outliers)
    return


@app.cell
def _(raw):
    negatives=(raw["Qty"] < 0).sum()
    print(negatives)
    return


@app.cell
def _(raw):
    null_rows = raw[raw[["Brand", "Range", "Style"]].isnull().any(axis=1)]
    null_rows
    return


@app.cell
def _(raw):
    neg = raw[raw["Qty"] < 0]
    neg["Qty"].describe()
    neg.sort_values("Qty").head(10)
    return (neg,)


@app.cell
def _(neg):
    neg["Qty"].describe()
    return


@app.cell
def _(neg):
    neg["Customer"].value_counts()
    neg["Site"].value_counts()
    return


@app.cell
def _(raw):
    raw.groupby("Site")["Customer"].nunique().describe()
    return


@app.cell
def _(temp):
    temp[["Date", "year", "month"]].head(10)
    return


@app.cell
def _(raw):
    raw["Date"].dt.day.max()
    return


@app.cell
def _(fig, pd, plt, raw):
    # parse with the exact format - no guessing allowed
    dates = pd.to_datetime(raw["Date"], format="%m/%d/%Y")
    temp = raw.assign(year=dates.dt.year, month=dates.dt.month)

    print("raw Date samples:", raw["Date"].unique()[:5])
    print("parsed months:", temp["month"].unique())

    # build the pivot
    pivot = temp.pivot_table(index="month", columns="year", values="Qty", aggfunc="sum")
    pivot.columns = pivot.columns.astype(str)

    # four bars, one shared scale, one figure
    y_max = pivot.max().max() * 1.05
    fig3, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax3, yr in zip(axes.flat, ["2022", "2023", "2024", "2025"]):
        ax3.bar(pivot.index, pivot[yr])
        ax3.set_title(yr)
        ax3.set_xlabel("Month")
        ax3.set_ylabel("Total Qty")
        ax3.set_ylim(0, y_max)
    fig.tight_layout()
    fig
    return (temp,)


@app.cell
def _(raw):
    raw.head()
    return


@app.cell
def _(raw):
    print(raw["Date"].dtype)
    raw["Date"].head(10)
    return


@app.cell
def _(raw):
    is_numeric = raw["Date"].astype(str).str.match(r'^\d+$')
    print("numeric-looking:", is_numeric.sum())
    print("text-looking:", (~is_numeric).sum())
    raw.loc[is_numeric, "source_sheet"].value_counts()
    return


@app.cell
def _(raw):
    raw.groupby("date_source_type")["Date"].apply(lambda s: s.dt.day.max())
    return


@app.cell
def _(raw):
    raw["Date"].dtype
    raw["Date"].head()
    return


@app.cell
def _(pd, raw):
    from datetime import datetime

    def classify_and_parse(val):
        if isinstance(val, datetime):
            return pd.to_datetime(val), "already_datetime"
        s = str(val)
        if s.isdigit():
            return pd.to_datetime(int(s), unit="D", origin="1899-12-30"), "excel_serial"
        return pd.to_datetime(s, format="%d/%m/%Y"), "text_date"          # day-first now, not month-first

    parsed = raw["Date"].apply(classify_and_parse)
    raw["Date"] = parsed.apply(lambda t: t[0])
    raw["date_source_type"] = parsed.apply(lambda t: t[1])

    raw["year"] = raw["Date"].dt.year
    raw["month"] = raw["Date"].dt.day

    print("month values:", sorted(raw["month"].unique()))
    return


@app.cell
def _(raw):
    raw.sample(10)
    return


@app.cell
def _(plt, raw):
    pivot2 = raw.pivot_table(index="month", columns="year", values="Qty", aggfunc="sum")
    pivot2.columns = pivot2.columns.astype(str)

    y_max2 = pivot2.max().max() * 1.05

    fig4, axes4 = plt.subplots(3, 4, figsize=(16, 10))
    for ax4, month in zip(axes4.flat, pivot2.index):
        ax4.bar(pivot2.columns, pivot2.loc[month])
        ax4.set_title(f"Month {month}")
        ax4.set_ylim(0, y_max2)
    fig4.tight_layout()
    fig4
    return


@app.cell
def _(raw):
    raw.pivot_table(index="month", columns="year", values="Qty", aggfunc="count")
    return


app._unparsable_cell(
    r"""
    +
    """,
    name="_"
)


@app.cell
def _(raw):
    raw.groupby("date_source_type")["Date"].apply(lambda s: s.dt.day.max())
    return


@app.cell
def _(raw):
    raw.sample(10)
    return


@app.cell
def _(raw):
    raw[["Customer", "Site", "Part", "Category", "Brand", "Range", "Style"]].nunique()
    return


@app.cell
def _(raw):
    raw.sort_values("Qty").head(10)
    return


@app.cell
def _(raw):
    raw.sort_values("Qty", ascending=False).head(10)
    return


@app.cell
def _(raw):
    raw["Date"].nunique()
    return


@app.cell
def _(raw):
    raw.groupby(["Site"]).size().value_counts()
    return


@app.cell
def _(files, pd):
    fresh = pd.read_excel(files[0], sheet_name=0, nrows=20)
    print(fresh["Date"])
    print(fresh["Date"].dtype)
    fresh.sample(20)
    return


@app.cell
def _(raw):
    raw.sample(40)
    return


@app.cell
def _(raw):
    raw.sample(10)
    return


@app.cell
def _(raw):
    ((raw.groupby(["Part","month","Site"]).size())>1).sum()
    return


@app.cell
def _(raw):
    negative=raw[raw["Qty"]<0]
    return (negative,)


@app.cell
def _(negative):
    (negative.groupby(["Part","month","Site"]).size()).sum()

    return


@app.cell
def _(negative, raw):
    keys = ["Part", "month", "Site"]

    raw_counts = raw.groupby(keys).size()

    negative_counts = negative.groupby(keys).size()

    # Groups in raw that have more than one row
    multi_groups = raw_counts[raw_counts > 1].index

    # Negative groups that are not in those multi-row groups
    uncovered = negative_counts.loc[
        ~negative_counts.index.isin(multi_groups)
    ]

    print("Uncovered negative rows:", uncovered.sum())
    print("Uncovered negative groups:", len(uncovered))
    return


@app.cell
def _(negative, raw):
    keys2 = ["Customer", "Site", "Part", "Date"]

    has_positive = raw[raw["Qty"] > 0].groupby(keys2).size()
    covered = negative.set_index(keys2).index.isin(has_positive.index)

    print("Negative rows WITH a same-day positive companion:", covered.sum())
    print("Negative rows WITHOUT one (isolated):", (~covered).sum())
    return


@app.cell
def _(negative, raw):
    keys_loose = ["Customer", "Site", "Part"]
    ever_positive = raw[raw["Qty"] > 0].groupby(keys_loose).size()
    covered_loose = negative.set_index(keys_loose).index.isin(ever_positive.index)

    print("Returns where this customer DID buy this item at this site at some point:", covered_loose.sum())
    print("Returns with NO matching purchase at all:", (~covered_loose).sum())
    return


@app.cell
def _(raw):
    print("Distinct Parts (item level):", raw["Part"].nunique())
    print("Distinct Ranges:", raw["Range"].nunique())
    return


@app.cell
def _(raw):
    raw.dtypes[["Customer","Site","Category","Brand","Part","Range"]] 
    return


@app.cell
def _():
    id_cols = ["Customer", "Site", "Category", "Brand"]
    return (id_cols,)


@app.cell
def _(id_cols, raw):
    raw[id_cols]=raw[id_cols].astype(str)
    return


@app.cell
def _(raw):
    raw.sample(10)
    return


@app.cell
def _(raw):
    raw["gross_qty"] = raw["Qty"].clip(lower=0)
    raw["return_qty"] = (-raw["Qty"]).clip(lower=0)
    return


@app.cell
def _(raw):
    raw.sample(10)
    return


@app.cell
def _(raw):
    import numpy as np
    monthly_demand = raw.groupby(["Site", "Range", "year", "month", "Brand"]).agg(
        gross_qty=("gross_qty", "sum"),
        return_qty=("return_qty", "sum")
    ).reset_index()

    assert monthly_demand.duplicated(subset=["Site", "Range", "year", "month", "Brand"]).sum() == 0
    print(f"{len(raw):,} transaction rows → {len(monthly_demand):,} Site×Range×month rows")


    keys4 = ["Site", "Brand", "Range"]


    # ============================================================
    # SECTION 6 — time_idx: one global month-counter
    # ============================================================
    ym = monthly_demand["year"] * 12 + monthly_demand["month"]
    ym_min = ym.min()
    monthly_demand["time_idx"] = ym - ym_min


    # ============================================================
    # SECTION 7 — GAP-FILL: one row per month per Site+Brand+Range
    # ============================================================
    global_max = monthly_demand["time_idx"].max()

    def fill_months(g):
        site, brand, rng = g.name          # captured FIRST - g.name is lost the moment g gets reassigned below
        full_idx = range(g["time_idx"].min(), global_max + 1)
        g = g.set_index("time_idx")
        g["is_filled"] = False
        g = g.reindex(full_idx)
        g["is_filled"] = g["is_filled"].fillna(True)
        g[["gross_qty", "return_qty"]] = g[["gross_qty", "return_qty"]].fillna(0)   # placeholder - corrected for reporting gaps below
        g["Site"], g["Brand"], g["Range"] = site, brand, rng
        return g

    monthly_demand = (
        monthly_demand.groupby(keys4, group_keys=False)
        .apply(fill_months, include_groups=False)
        .reset_index()
        .rename(columns={"index": "time_idx"})
    )

    ym_full = monthly_demand["time_idx"] + ym_min
    monthly_demand["year"] = ym_full // 12
    monthly_demand["month"] = ym_full % 12
    dec_mask = monthly_demand["month"] == 0
    monthly_demand.loc[dec_mask, "year"] -= 1
    monthly_demand.loc[dec_mask, "month"] = 12


    # ============================================================
    # SECTION 8 — SEASONALITY: month as a position on a circle
    # ============================================================
    monthly_demand["month_sin"] = np.sin(2 * np.pi * monthly_demand["month"] / 12)
    monthly_demand["month_cos"] = np.cos(2 * np.pi * monthly_demand["month"] / 12)


    # ============================================================
    # SECTION 8B — REPORTING GAP: tell a real zero-sales month apart from a missing export
    # ============================================================
    # A sandwiched gap only means "genuinely zero" if it's a FEW groups quiet that
    # month - if almost every group is "filled" at once, that's the whole export
    # missing, not real demand. Must run BEFORE Section 9, so shift()/rolling()
    # inherit NaN naturally instead of averaging in a fabricated 0.
    REPORTING_GAP_THRESHOLD = 0.9   # tune against the printed fill-rate below if needed

    fill_rate_by_month = monthly_demand.groupby("time_idx")["is_filled"].mean()
    reporting_gap_months = fill_rate_by_month[fill_rate_by_month > REPORTING_GAP_THRESHOLD].index

    is_reporting_gap = monthly_demand["time_idx"].isin(reporting_gap_months) & monthly_demand["is_filled"]
    monthly_demand.loc[is_reporting_gap, ["gross_qty", "return_qty"]] = np.nan

    print("fill rate by month (tail):")
    print(fill_rate_by_month.tail(15))
    print("flagged as reporting-gap months:", list(reporting_gap_months))


    # ============================================================
    # SECTION 9 — LIFECYCLE + HISTORY FEATURES
    # ============================================================
    monthly_demand = monthly_demand.sort_values(keys4 + ["time_idx"])

    monthly_demand["range_age_months"] = (
        monthly_demand["time_idx"] - monthly_demand.groupby(keys4)["time_idx"].transform("min")
    )

    g = monthly_demand.groupby(keys4)["gross_qty"]
    monthly_demand["lag_1_gross_qty"]  = g.shift(1)
    monthly_demand["rolling_mean_3"]   = g.transform(lambda s: s.shift(1).rolling(3).mean())
    monthly_demand["rolling_mean_6"]   = g.transform(lambda s: s.shift(1).rolling(6).mean())
    monthly_demand["lag_12_gross_qty"] = g.shift(12)

    total = monthly_demand["gross_qty"] + monthly_demand["return_qty"]
    monthly_demand["return_rate"] = (monthly_demand["return_qty"] / total).where(total > 0)
    monthly_demand["return_rate_prev_month"] = monthly_demand.groupby(keys4)["return_rate"].shift(1)


    # ============================================================
    # SECTION 10 — TRAINING SET: drop only rows where the target itself is unknown
    # ============================================================
    # ============================================================
    # SECTION 10 — TRAINING SET: drop only rows where the target itself is unknown
    # ============================================================
    # data_complete_through = monthly_demand.loc[~monthly_demand["is_filled"], "time_idx"].max()   # informational only now
    training_data = monthly_demand.dropna(subset=["gross_qty"])

    dropped_rows = monthly_demand[monthly_demand["gross_qty"].isna()]   # the ones training_data left behind

    print(f"monthly_demand: {len(monthly_demand):,} rows | training_data: {len(training_data):,} rows | dropped: {len(dropped_rows):,}")
    # print(f"last fully-reported time_idx (for reference): {data_complete_through}")
    print()
    print("dropped rows by year/month (should line up with reporting_gap_months from Section 8B):")
    print(dropped_rows.groupby(["year", "month"]).size())
    print()
    print("dropped rows by Site×Brand×Range (which groups actually lost rows, and how many):")
    print(dropped_rows.groupby(keys4).size().sort_values(ascending=False).head(10))
    return dropped_rows, global_max, keys4, monthly_demand, training_data


@app.cell
def _(monthly_demand):
    monthly_demand.sample(20)
    return


@app.cell
def _(dropped_rows):
    dropped_rows.info()
    return


@app.cell
def _(training_data):
    training_data.info()
    return


@app.cell
def _(monthly_demand):
    monthly_demand.groupby(["year","month"])["is_filled"].mean().to_string()
    return


@app.cell
def _(monthly_demand):
    monthly_demand[(monthly_demand["year"]==2025) & (monthly_demand["month"].isin([1,2,3]))]["is_filled"].mean()
    return


@app.cell
def _(global_max, keys4, monthly_demand):
    monthly_demand["is_filled"] = monthly_demand["is_filled"].astype(bool)
    real = monthly_demand[~monthly_demand["is_filled"]]   # just the 12,517 real rows, recovered via the flag

    lifespan = real.groupby(keys4)["time_idx"].agg(["min", "count"])
    lifespan["expected_months"] = global_max - lifespan["min"] + 1   # how many months SHOULD exist, start to dataset-end
    lifespan["missing_months"] = lifespan["expected_months"] - lifespan["count"]

    print("Missing months, computed independently:", lifespan["missing_months"].sum())
    print("is_filled count, from the gap-fill itself:", monthly_demand["is_filled"].sum())
    return


@app.cell
def _(monthly_demand):

    # Calculate the proportion of gap-filled rows for each month
    check = (
        monthly_demand[
            (monthly_demand["year"] == 2025)
            & (monthly_demand["month"].isin([1, 2, 3]))
        ]
        .groupby(["year", "month"])["is_filled"]
        .agg(["mean", "count"])
    )

    check["mean"] = check["mean"].round(3)

    print(check.to_string())

    return


@app.cell
def _(monthly_demand):
    subset = monthly_demand[(monthly_demand["year"] == 2025) & (monthly_demand["month"].isin([1, 2, 3]))]

    print("Total rows for Jan-Mar 2025:", len(subset))
    print("Of those, is_filled=True:", subset["is_filled"].sum())
    print("Of those, is_filled=False (real):", (~subset["is_filled"]).sum())
    print()
    subset.groupby("month")["is_filled"].agg(["sum", "count", "mean"])
    return


@app.cell
def _(monthly_demand):
    monthly_demand[(monthly_demand["year"]==2025) & (monthly_demand["month"].isin([1,2,3])) & (~monthly_demand["is_filled"])].head(10)
    return


@app.cell
def _(monthly_demand, training_data):
    print("Rows with gross_qty still NaN:", monthly_demand["gross_qty"].isna().sum())
    print("training_data rows == monthly_demand rows:", len(training_data) == len(monthly_demand))
    print()
    print("Real date range:", (monthly_demand["year"].astype(str) + "-" + monthly_demand["month"].astype(str).str.zfill(2)).agg(["min","max"]))
    return


@app.cell
def _(monthly_demand):
    monthly_demand.info()
    return


@app.cell
def _(monthly_demand):
    monthly_demand["display_id"] = (
        monthly_demand["Site"] + "_" + monthly_demand["Brand"] + "_" + monthly_demand["Range"]
        + "_" + monthly_demand["year"].astype(str) + "-" + monthly_demand["month"].astype(str).str.zfill(2)
    )
    return


@app.cell
def _(monthly_demand):
    monthly_demand.sample(10)
    return


@app.cell
def _(monthly_demand):
    assert (monthly_demand["lag_1_gross_qty"].isna()  == (monthly_demand["range_age_months"] < 1)).all()
    assert (monthly_demand["rolling_mean_3"].isna()   == (monthly_demand["range_age_months"] < 3)).all()
    assert (monthly_demand["rolling_mean_6"].isna()   == (monthly_demand["range_age_months"] < 6)).all()
    assert (monthly_demand["lag_12_gross_qty"].isna() == (monthly_demand["range_age_months"] < 12)).all()
    print("All lag/rolling NaN patterns match range_age_months exactly")
    return


@app.cell
def _(Path, monthly_demand, training_data):
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    monthly_demand.to_pickle(processed_dir / "monthly_demand.pkl")
    training_data.to_pickle(processed_dir / "training_data.pkl")

    print(f"Saved monthly_demand: {len(monthly_demand):,} rows")
    print(f"Saved training_data:  {len(training_data):,} rows")
    return (processed_dir,)


@app.cell
def _(monthly_demand, pd, processed_dir):
    check2 = pd.read_pickle(processed_dir / "monthly_demand.pkl")
    assert len(check2) == len(monthly_demand)
    assert check2["is_filled"].dtype == bool
    print("Reload verified clean")
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
