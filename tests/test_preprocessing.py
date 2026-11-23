import pandas as pd
import pytest

from src.data_preprocessing import (auto_map_columns, clean_dataset,
                                    normalize_description, standardize_category)


def test_normalize_strips_boilerplate_and_punctuation():
    assert normalize_description("Transaction at Nature's Basket") == "natures basket"
    assert normalize_description("Transaction at H&M") == "h and m"


def test_normalize_handles_empty_and_none():
    assert normalize_description(None) == ""
    assert normalize_description("   ") == ""
    assert normalize_description(float("nan")) == ""


def test_standardize_category_aliases():
    assert standardize_category("groceries") == "Grocery"
    assert standardize_category("  Food ") == "Food"


def _frame(rows):
    return pd.DataFrame(rows, columns=["date", "description", "amount", "category"])


def test_clean_reports_every_dropped_row():
    df = _frame([
        ["2025-01-01", "Transaction at KFC", 250, "Food"],            # ok
        ["2025-01-02", "", 100, "Food"],                               # empty description
        ["2025-01-03", "Transaction at Ola", -50, "Transport"],        # negative amount
        ["not a date", "Transaction at Ola", 80, "Transport"],         # bad date
        ["2025-01-05", "Transaction at Ola", "abc", "Transport"],      # bad amount
        ["2025-01-06", "Transaction at Apollo Pharmacy", 90, "Healthcare"],  # not in 'five'
        ["2025-01-01", "Transaction at KFC", 250, "Food"],            # exact duplicate
    ])
    clean, report = clean_dataset(df, mode="five")
    assert len(clean) == 1
    assert report["rows_in"] == 7 and report["rows_out"] == 1
    assert sum(report["dropped"].values()) == 6


def test_five_mode_merges_shopping_types():
    df = _frame([
        ["2025-01-01", "Transaction at Myntra", 100, "Online Shopping"],
        ["2025-01-02", "Transaction at Zara", 200, "Clothing"],
        ["2025-01-03", "Transaction at Croma", 300, "Electronics"],
    ])
    clean, _ = clean_dataset(df, mode="five")
    assert set(clean["category"]) == {"Shopping"}


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        clean_dataset(_frame([]), mode="nope")


def test_auto_map_columns_finds_narration_and_debit():
    df = pd.DataFrame(columns=["Txn Date", "Narration", "Debit"])
    assert auto_map_columns(df) == {"date": "Txn Date", "description": "Narration", "amount": "Debit"}
