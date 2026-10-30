"""
prediction.py - loads the saved model and turns patient inputs into a prediction.
"""
import warnings

import joblib
import pandas as pd
import sklearn

from src.preprocessing import FEATURE_COLUMNS, MODEL_PATH, ZERO_AS_MISSING_COLUMNS

# (min, max, label) - realistic input ranges used by the app's validation.
# SkinThickness / Insulin may be 0, meaning "not measured" (imputed by the pipeline).
INPUT_RULES = {
    "Pregnancies": (0, 20, "Pregnancies"),
    "Glucose": (40, 300, "Glucose (mg/dL)"),
    "BloodPressure": (20, 200, "Diastolic blood pressure (mm Hg)"),
    "SkinThickness": (0, 100, "Skin-fold thickness (mm)"),
    "Insulin": (0, 900, "Insulin (mu U/ml)"),
    "BMI": (10.0, 70.0, "BMI (kg/m^2)"),
    "DiabetesPedigreeFunction": (0.05, 2.5, "Diabetes pedigree function"),
    "Age": (21, 90, "Age (years)"),
}
OPTIONAL_ZERO = {"SkinThickness", "Insulin"}


def load_model_bundle(path=MODEL_PATH) -> dict:
    """Load the saved bundle; raise readable errors instead of stack traces."""
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}.\n"
            "Train it first with:  python -m src.train_model")
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load the model ({exc}). Re-train with: python -m src.train_model") from exc
    if not {"pipeline", "feature_columns"}.issubset(bundle):
        raise RuntimeError("Model file has an unexpected format. Re-train the model.")
    if bundle["feature_columns"] != FEATURE_COLUMNS:
        raise RuntimeError("Feature order in the model differs from the code. Re-train the model.")
    if bundle.get("sklearn_version") != sklearn.__version__:
        warnings.warn(
            f"Model was trained with scikit-learn {bundle.get('sklearn_version')} but "
            f"{sklearn.__version__} is installed. Re-train if you see errors.")
    return bundle


def validate_inputs(values: dict) -> list:
    """Return a list of human-readable problems (empty list = inputs are valid)."""
    errors = []
    for feature in FEATURE_COLUMNS:
        if feature not in values or values[feature] is None:
            errors.append(f"{feature} is missing.")
            continue
        try:
            v = float(values[feature])
        except (TypeError, ValueError):
            errors.append(f"{INPUT_RULES[feature][2]} must be a number.")
            continue
        lo, hi, label = INPUT_RULES[feature]
        if not (lo <= v <= hi):
            errors.append(f"{label} must be between {lo} and {hi}.")
    return errors


def predict_risk(bundle: dict, values: dict) -> dict:
    """Run the SAME pipeline used in training on one patient record."""
    errors = validate_inputs(values)
    if errors:
        raise ValueError(" ".join(errors))

    # Build a one-row DataFrame in the exact training column order.
    row = pd.DataFrame([[float(values[c]) for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    pipe = bundle["pipeline"]
    predicted_class = int(pipe.predict(row)[0])
    p_pos = float(pipe.predict_proba(row)[0, 1])

    # Warn (don't block) if an input lies outside what the model saw in training.
    outside = []
    for c, (lo, hi) in bundle.get("feature_ranges", {}).items():
        v = float(values[c])
        if c in ZERO_AS_MISSING_COLUMNS and v == 0:
            continue  # 0 means "unknown" for these columns
        if v < lo or v > hi:
            outside.append(c)
    return {"predicted_class": predicted_class, "probability_positive": p_pos,
            "probability_negative": 1 - p_pos, "outside_training_range": outside,
            "model_name": bundle.get("model_name", "model")}
