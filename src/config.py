"""Central settings shared by preprocessing, training and (later) the app."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = ROOT / "data" / "raw" / "personal_finance_dataset_cleaned_flagged.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

MODEL_PATH = MODELS_DIR / "best_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2

DATASET_NAME = "personal_finance_dataset_cleaned_flagged.csv (user supplied)"
DATASET_NOTE = (
    "Provenance not verified. Every description follows the template "
    "'Transaction at <Merchant>', classes are almost perfectly balanced and each "
    "merchant maps to exactly one category, so the data appears synthetic. "
    "Do not describe it as real bank transactions in the report."
)

# Words/phrases that are boilerplate around the merchant name, not information.
BOILERPLATE_PREFIXES = ("transaction at", "payment to", "paid to", "purchase at")

# Lower-case spelling variants -> canonical category name.
CATEGORY_ALIASES = {
    "groceries": "Grocery",
    "grocery": "Grocery",
    "food & dining": "Food",
    "food and dining": "Food",
    "shopping": "Online Shopping",
    "online shopping": "Online Shopping",
    "health": "Healthcare",
    "medical": "Healthcare",
    "utilities": "Bills",
}

# Label modes.
#   "five": the five target categories of the project. Several native categories are
#           merged into Shopping; native categories not listed here are DROPPED
#           (and reported), not silently relabelled.
#   "full": all native categories of the dataset are kept as they are.
CATEGORY_MAPS = {
    "five": {
        "Food": "Food",
        "Transport": "Transport",
        "Online Shopping": "Shopping",
        "Clothing": "Shopping",
        "Electronics": "Shopping",
        "Bills": "Bills",
        "Entertainment": "Entertainment",
    },
    "full": None,  # None = keep native labels
}

# Column-name aliases used to map a user's file onto the required fields.
COLUMN_ALIASES = {
    "date": ["date", "transaction date", "txn date", "value date", "posting date"],
    "description": ["description", "narration", "details", "particulars",
                    "transaction details", "remarks", "merchant"],
    "amount": ["amount", "debit", "withdrawal", "withdrawal amt.", "txn amount"],
    "category": ["category", "label", "expense category"],
}
