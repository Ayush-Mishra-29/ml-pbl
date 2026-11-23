"""Loading, validating, cleaning and exploring the expense dataset.

Run from the project root:
    python -m src.data_preprocessing --mode five
"""
import argparse
import json
import re

import matplotlib

matplotlib.use("Agg")  # no GUI needed; we only save PNG files
import matplotlib.pyplot as plt
import pandas as pd

from src import config

REQUIRED_COLUMNS = ["date", "description", "amount", "category"]


# --------------------------------------------------------------------------
# Text normalisation (used BOTH in training and prediction)
# --------------------------------------------------------------------------
def normalize_description(text) -> str:
    """Lower-case, strip boilerplate prefixes and punctuation, keep merchant words.

    'Transaction at Nature's Basket' -> 'natures basket'
    'Transaction at H&M'             -> 'h and m'
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text).lower().strip()
    s = s.replace("&", " and ").replace("'", "")
    for prefix in config.BOILERPLATE_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    s = re.sub(r"[^a-z0-9\s]", " ", s)   # punctuation -> space
    s = re.sub(r"\d{4,}", " ", s)         # long digit runs (ref numbers) carry no meaning
    s = re.sub(r"\s+", " ", s).strip()
    return s


def standardize_category(label) -> str:
    """Trim whitespace, unify case, and resolve known spelling variants."""
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return ""
    s = re.sub(r"\s+", " ", str(label)).strip()
    return config.CATEGORY_ALIASES.get(s.lower(), s.title() if s.islower() or s.isupper() else s)


# --------------------------------------------------------------------------
# Column mapping (reused by the bulk-import page later)
# --------------------------------------------------------------------------
def auto_map_columns(df: pd.DataFrame) -> dict:
    """Guess {required_field: actual_column}. Missing fields are simply absent."""
    lowered = {c.lower().strip(): c for c in df.columns}
    mapping = {}
    for field, aliases in config.COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                mapping[field] = lowered[alias]
                break
    return mapping


def load_raw(path=config.RAW_DATA_PATH, column_map: dict | None = None) -> pd.DataFrame:
    """Read a CSV and return a frame with columns date/description/amount/category."""
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("The file contains no rows.")
    mapping = column_map or auto_map_columns(df)
    missing = [f for f in ("description", "amount") if f not in mapping]
    if missing:
        raise ValueError(f"Could not find columns for: {missing}. Found: {list(df.columns)}")
    out = pd.DataFrame({field: df[col] for field, col in mapping.items()})
    for field in REQUIRED_COLUMNS:
        if field not in out.columns:
            out[field] = pd.NA
    return out[REQUIRED_COLUMNS]


# --------------------------------------------------------------------------
# Cleaning with a full audit trail
# --------------------------------------------------------------------------
def clean_dataset(df: pd.DataFrame, mode: str = "five"):
    """Return (clean_df, report). Every dropped row is counted with a reason.

    Convention: expenses are POSITIVE amounts (the source file is all 'Debit').
    """
    if mode not in config.CATEGORY_MAPS:
        raise ValueError(f"mode must be one of {list(config.CATEGORY_MAPS)}")

    df = df.copy()
    report = {"rows_in": int(len(df)), "mode": mode, "dropped": {}}

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["description_raw"] = df["description"].astype("string")
    df["description_clean"] = df["description_raw"].map(normalize_description)
    df["category_native"] = df["category"].map(standardize_category)

    def drop(mask, reason):
        n = int(mask.sum())
        if n:
            report["dropped"][reason] = n
        return df.loc[~mask]

    df = drop(df["description_clean"].eq(""), "empty or unusable description")
    df = drop(df["amount"].isna(), "amount not numeric")
    df = drop(df["amount"] <= 0, "amount zero or negative")
    df = drop(df["date"].isna(), "date missing or unparseable")
    df = drop(df["category_native"].eq(""), "category missing")

    mapping = config.CATEGORY_MAPS[mode]
    if mapping is None:
        df["category"] = df["category_native"]
    else:
        unmapped = ~df["category_native"].isin(mapping)
        report["unmapped_native_categories"] = (
            df.loc[unmapped, "category_native"].value_counts().to_dict()
        )
        df = drop(unmapped, f"native category not part of '{mode}' mode")
        df["category"] = df["category_native"].map(mapping)

    dup = df.duplicated(subset=["date", "description_clean", "amount", "category"], keep="first")
    df = drop(dup, "exact duplicate (date, description, amount, category)")

    df = df.reset_index(drop=True)
    # Used only for merchant-grouped evaluation: same normalised text = same merchant.
    df["merchant_key"] = df["description_clean"]

    report["rows_out"] = int(len(df))
    report["class_counts"] = df["category"].value_counts().to_dict()
    counts = df["category"].value_counts()
    report["imbalance_ratio_max_to_min"] = round(float(counts.max() / counts.min()), 2)
    report["unique_merchant_keys"] = int(df["merchant_key"].nunique())
    return df, report


# --------------------------------------------------------------------------
# Exploratory plots
# --------------------------------------------------------------------------
def make_eda_plots(raw: pd.DataFrame, clean: pd.DataFrame, outdir=config.FIGURES_DIR):
    outdir.mkdir(parents=True, exist_ok=True)
    saved = []

    def save(fig, name):
        path = outdir / name
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        saved.append(str(path))

    counts = clean["category"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot.bar(ax=ax, color="#3b6ea5")
    ax.set_title("Transactions per category (cleaned data)")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    save(fig, "category_distribution.png")

    miss = raw.isna().sum()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    miss.plot.barh(ax=ax, color="#b5533c")
    ax.set_title("Missing values per required column (raw data)")
    ax.set_xlabel("Missing cells")
    save(fig, "missing_values.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(clean["amount"], bins=50, color="#3b6ea5")
    ax.set_yscale("log")
    ax.set_title("Transaction amount distribution (log y-axis)")
    ax.set_xlabel("Amount (INR)")
    save(fig, "amount_distribution.png")

    monthly = clean.groupby(clean["date"].dt.to_period("M")).size()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(monthly.index.astype(str), monthly.values, marker="o", color="#3b6ea5")
    ax.set_title("Monthly transaction volume")
    ax.set_ylabel("Transactions")
    ax.tick_params(axis="x", rotation=45)
    save(fig, "monthly_volume.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    clean.boxplot(column="amount", by="category", ax=ax, rot=30)
    ax.set_yscale("log")
    ax.set_title("Amount by category (log y-axis)")
    fig.suptitle("")
    save(fig, "amount_by_category.png")
    return saved


# --------------------------------------------------------------------------
def run(mode="five", path=config.RAW_DATA_PATH):
    raw = load_raw(path)
    print(f"Loaded {raw.shape[0]} rows x {raw.shape[1]} mapped columns from {path}")
    print("Missing values:\n", raw.isna().sum().to_string())
    clean, report = clean_dataset(raw, mode=mode)
    figs = make_eda_plots(raw, clean)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = config.PROCESSED_DIR / f"clean_{mode}.csv"
    clean.to_csv(out_csv, index=False)
    with open(config.PROCESSED_DIR / f"preprocessing_report_{mode}.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))
    print(f"Saved cleaned data -> {out_csv}")
    print(f"Saved {len(figs)} figures -> {config.FIGURES_DIR}")
    return clean, report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(config.CATEGORY_MAPS), default="five")
    ap.add_argument("--data", default=str(config.RAW_DATA_PATH))
    args = ap.parse_args()
    run(args.mode, args.data)
