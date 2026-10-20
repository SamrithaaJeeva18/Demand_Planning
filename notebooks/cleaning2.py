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
    from datetime import datetime
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

    # Date comes out of the sheets in three different shapes - handle all three, then extract year/month
    def parse_date(val):
        if isinstance(val, datetime):
            return pd.to_datetime(val)
        s = str(val)
        if s.isdigit():
            return pd.to_datetime(int(s), unit="D", origin="1899-12-30")   # Excel serial number
        return pd.to_datetime(s, format="%d/%m/%Y")                         # day-first, day always 1

    raw["Date"] = raw["Date"].apply(parse_date)
    raw["year"] = raw["Date"].dt.year
    raw["month"] = raw["Date"].dt.Date

    print(f"Merged {len(files)} files, {len(dfs)} sheets total → {len(raw):,} rows")
    print("day values (should be only [1]):", raw["Date"].dt.day.unique())
    print("month values:", sorted(raw["month"].unique()))
    return datetime, pd, raw


@app.cell
def _(raw):
    raw.sample(30)
    return


@app.cell
def _(raw):
    raw.groupby(["Customer", "Site", "Part", "Date"]).size().value_counts()
    return


@app.cell
def _(raw):
    raw["month"].value_counts().sort_index()
    return


@app.cell
def _(datetime, pd, raw):


    def classify(val):
        if isinstance(val, datetime):
            return pd.to_datetime(val), "already_datetime"
        s = str(val)
        if s.isdigit():
            return pd.to_datetime(int(s), unit="D", origin="1899-12-30"), "excel_serial"
        return pd.to_datetime(s, format="%d/%m/%Y"), "text_date"

    parsed = raw["Date"].apply(classify)
    raw["Date"] = parsed.apply(lambda t: t[0])
    raw["date_source_type"] = parsed.apply(lambda t: t[1])

    print(raw["date_source_type"].value_counts())
    print(raw.groupby("date_source_type")["Date"].apply(lambda s: s.dt.day.max()))
    return


@app.cell
def _(raw):
    raw.sample(50)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
