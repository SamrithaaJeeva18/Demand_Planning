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
    import sys
    import pandas as pd

    project_root = Path.cwd()
    if not (project_root / "models").is_dir():
        project_root = project_root.parent

    # Put this project first and clear any unrelated installed package named `models`.
    sys.path.insert(0, str(project_root))
    for module_name in tuple(sys.modules):
        if module_name == "models" or module_name.startswith("models."):
            del sys.modules[module_name]

    from models.windows import WindowBuilder, WindowedForecaster

    # Load the prepared monthly training data from the project root.
    data_path = project_root / "data" / "processed" / "training_data.pkl"
    panel = pd.read_pickle(data_path).dropna(subset=["gross_qty"])
    print(f"Loaded {len(panel):,} rows from {data_path}")
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
