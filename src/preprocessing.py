"""
preprocessing.py - dataset loading, quality checks and the preprocessing pipeline.

Everything that touches raw data lives here so that TRAINING and PREDICTION use
exactly the same cleaning steps (this is what prevents "training/serving skew").
"""
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# ----------------------------------------------------------------------------
# Paths and constants
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "diabetes.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "diabetes_model.pkl"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
FIGURES_DIR = PROJECT_ROOT / "assets" / "figures"

RANDOM_STATE = 42   # one seed everywhere -> reproducible results
TEST_SIZE = 0.20

# The ORDER of this list is the order of columns used in training AND prediction.
FEATURE_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
TARGET_COLUMN = "Outcome"  # 0 = non-diabetic, 1 = diabetic

# In these columns a value of 0 is physiologically impossible (e.g. BMI = 0),
# so a 0 really means "not recorded" and is treated as a missing value.
ZERO_AS_MISSING_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Public mirrors of the Pima Indians Diabetes dataset (tried in order).
DATASET_URLS = [
    "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
]

MANUAL_INSTRUCTIONS = (
    "Dataset not found at: {path}\n"
    "Download 'diabetes.csv' (Pima Indians Diabetes Database) from Kaggle:\n"
    "  https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database\n"
    "(original source: UCI / NIDDK) and save it as:\n  {path}\n"
    "It must have 768 rows and the 9 columns: "
    + ", ".join(FEATURE_COLUMNS + [TARGET_COLUMN])
)


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_dataset(path=DATA_PATH) -> pd.DataFrame:
    """Load the CSV and validate that it has the expected columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(MANUAL_INSTRUCTIONS.format(path=path))

    df = pd.read_csv(path)
    expected = FEATURE_COLUMNS + [TARGET_COLUMN]
    if not set(expected).issubset(df.columns):
        # Some mirrors ship the file WITHOUT a header row - retry with our names.
        df = pd.read_csv(path, header=None, names=expected)
    try:
        df = df[expected].apply(pd.to_numeric, errors="raise")
    except (ValueError, KeyError) as exc:
        raise ValueError(
            f"{path} does not look like the Pima Indians Diabetes CSV. "
            f"Expected columns: {expected}"
        ) from exc
    return df


def download_dataset(path=DATA_PATH) -> bool:
    """Try to download the dataset automatically. Returns True on success."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for url in DATASET_URLS:
        try:
            print(f"Trying to download dataset from {url} ...")
            urllib.request.urlretrieve(url, path)
            df = load_dataset(path)          # validates the file
            if len(df) == 768:
                print(f"Dataset saved to {path} ({len(df)} rows).")
                return True
            print(f"  Unexpected row count ({len(df)}), trying next source.")
        except Exception as exc:             # network error, bad file, ...
            print(f"  Failed: {exc}")
        if path.exists():
            path.unlink()
    return False


# ----------------------------------------------------------------------------
# Quality report (used by train_model.py, the notebook and the app)
# ----------------------------------------------------------------------------
def data_quality_report(df: pd.DataFrame) -> dict:
    """Missing values, duplicates, and 'hidden missing' zeros."""
    return {
        "n_records": int(df.shape[0]),
        "n_features": int(len(FEATURE_COLUMNS)),
        "missing_values": {c: int(v) for c, v in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "invalid_zero_counts": {c: int((df[c] == 0).sum()) for c in ZERO_AS_MISSING_COLUMNS},
        "class_counts": {int(k): int(v) for k, v in df[TARGET_COLUMN].value_counts().sort_index().items()},
    }


# ----------------------------------------------------------------------------
# Preprocessing pipeline
# ----------------------------------------------------------------------------
def zeros_to_nan(X) -> pd.DataFrame:
    """Replace impossible zeros with NaN so the imputer can fill them.

    Being a pipeline step, this runs identically at training and prediction time.
    """
    X = pd.DataFrame(X).copy()
    if list(X.columns) != FEATURE_COLUMNS:
        if X.shape[1] != len(FEATURE_COLUMNS):
            raise ValueError(f"Expected {len(FEATURE_COLUMNS)} features, got {X.shape[1]}")
        X.columns = FEATURE_COLUMNS
    X = X.astype(float)
    X[ZERO_AS_MISSING_COLUMNS] = X[ZERO_AS_MISSING_COLUMNS].replace(0, np.nan)
    return X


def build_pipeline(classifier, scale: bool) -> Pipeline:
    """zeros->NaN -> median imputation -> (optional scaling) -> classifier.

    Because everything is inside ONE Pipeline, imputer medians and scaler
    statistics are learned from the training folds only -> no data leakage.
    Scaling matters for Logistic Regression, not for tree models.
    """
    steps = [
        ("zeros_to_nan", FunctionTransformer(zeros_to_nan, validate=False)),
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def split_data(df: pd.DataFrame):
    """Stratified 80/20 split (keeps the diabetic/non-diabetic ratio in both parts)."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
