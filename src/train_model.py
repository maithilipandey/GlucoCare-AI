"""
train_model.py - trains, tunes, compares and saves the diabetes models.

Run from the project root:   python -m src.train_model

Design decisions (explained in README / VIVA_QUESTIONS):
  * Stratified 80/20 split, one fixed seed.
  * Both models live in a Pipeline (impute -> scale -> classifier), so nothing
    from the test set leaks into preprocessing.
  * Hyper-parameters are tuned with 5-fold stratified cross-validation on the
    TRAINING data only.
  * The FINAL MODEL is chosen from cross-validation results (not from the test
    set) using F2-score, which weights recall twice as much as precision,
    because a missed diabetic case (false negative) is the costlier error.
  * The held-out test set is used once, for the final honest evaluation.
"""
import json
import sys
from pathlib import Path

# Make `import src...` work no matter how the script is launched, and make the
# pickled pipeline refer to `src.preprocessing` (needed when the app loads it).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, make_scorer, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.evaluation import (compute_metrics, plot_confusion_matrix, plot_feature_importance,
                            plot_model_comparison, plot_roc_curves)
from src.preprocessing import (DATA_PATH, FEATURE_COLUMNS, FIGURES_DIR, MANUAL_INSTRUCTIONS,
                               METRICS_PATH, MODEL_PATH, RANDOM_STATE, TEST_SIZE, build_pipeline,
                               data_quality_report, download_dataset, load_dataset, split_data,
                               zeros_to_nan)

SCORING = {
    "f2": make_scorer(fbeta_score, beta=2),
    "recall": "recall", "precision": "precision", "f1": "f1",
    "roc_auc": "roc_auc", "accuracy": "accuracy",
}


def get_model_configs():
    """Pipelines + small hyper-parameter grids for the two classifiers."""
    lr = build_pipeline(
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        scale=True)
    rf = build_pipeline(
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        scale=False)
    return {
        "Logistic Regression": (lr, {"classifier__C": [0.01, 0.1, 1, 10]}),
        "Random Forest": (rf, {
            "classifier__n_estimators": [200, 400],
            "classifier__max_depth": [3, 5, 8, None],
            "classifier__min_samples_leaf": [1, 3, 5],
        }),
    }


def main():
    # 1. Load data (try to download if missing) ---------------------------------
    if not DATA_PATH.exists() and not download_dataset():
        print("\n" + MANUAL_INSTRUCTIONS.format(path=DATA_PATH))
        sys.exit(1)
    df = load_dataset()

    # 2. Data quality report -----------------------------------------------------
    report = data_quality_report(df)
    print("=== DATA QUALITY ===")
    print(f"Records: {report['n_records']}  Features: {report['n_features']}")
    print(f"Missing (NaN) values: {sum(report['missing_values'].values())}")
    print(f"Duplicate rows: {report['duplicate_rows']}")
    print(f"Impossible zeros (treated as missing): {report['invalid_zero_counts']}")
    print(f"Class counts (0=non-diabetic, 1=diabetic): {report['class_counts']}")

    # 3. Stratified split --------------------------------------------------------
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows (stratified, seed={RANDOM_STATE})")

    # 4. Tune each model with CV on the TRAINING set ----------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fitted, results = {}, {}
    for name, (pipe, grid) in get_model_configs().items():
        print(f"\nTuning {name} ...")
        gs = GridSearchCV(pipe, grid, scoring=SCORING, refit="f2", cv=cv, n_jobs=1)
        gs.fit(X_train, y_train)
        idx = gs.best_index_
        cv_scores = {k: float(gs.cv_results_[f"mean_test_{k}"][idx]) for k in SCORING}
        print(f"  best params: {gs.best_params_}")
        print(f"  CV  F2={cv_scores['f2']:.3f}  recall={cv_scores['recall']:.3f}  "
              f"AUC={cv_scores['roc_auc']:.3f}")

        # 5. One-time evaluation on the untouched test set ------------------------
        best = gs.best_estimator_
        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, 1]
        test = compute_metrics(y_test, y_pred, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        print(f"  TEST acc={test['accuracy']:.3f} prec={test['precision']:.3f} "
              f"recall={test['recall']:.3f} f1={test['f1']:.3f} auc={test['roc_auc']:.3f} "
              f"(false negatives={test['fn']})")
        fitted[name] = best
        results[name] = {
            "best_params": {k.replace("classifier__", ""): v for k, v in gs.best_params_.items()},
            "cv": cv_scores, "test": test,
            "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": test["roc_auc"]},
        }

    # 6. Select the final model using CV F2 (never the test set) ------------------
    selected = max(results, key=lambda n: results[n]["cv"]["f2"])
    others = [n for n in results if n != selected]
    sel = results[selected]
    reason = (
        f"{selected} was selected because it achieved the highest mean cross-validated F2-score "
        f"on the training data ({sel['cv']['f2']:.3f}"
        + "".join(f" vs {results[o]['cv']['f2']:.3f} for {o}" for o in others)
        + "). F2 weights recall twice as much as precision, which suits screening where missing a "
        f"diabetic case is costly. On the held-out test set it reached recall "
        f"{sel['test']['recall']:.3f} with {sel['test']['fn']} false negatives and ROC-AUC "
        f"{sel['test']['roc_auc']:.3f}."
    )
    print(f"\n=== FINAL MODEL: {selected} ===\n{reason}")

    # 7. Interpretation data ------------------------------------------------------
    rf_clf = fitted["Random Forest"].named_steps["classifier"]
    importance = dict(zip(FEATURE_COLUMNS, map(float, rf_clf.feature_importances_)))
    lr_clf = fitted["Logistic Regression"].named_steps["classifier"]
    coefficients = dict(zip(FEATURE_COLUMNS, map(float, lr_clf.coef_[0])))

    # 8. Save model bundle (pipeline + metadata) ----------------------------------
    cleaned_train = zeros_to_nan(X_train)
    ranges = {c: [float(cleaned_train[c].min()), float(cleaned_train[c].max())] for c in FEATURE_COLUMNS}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": fitted[selected], "model_name": selected,
        "feature_columns": FEATURE_COLUMNS, "feature_ranges": ranges,
        "sklearn_version": sklearn.__version__,
    }, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # 9. Save metrics (read by the Streamlit app) + figures + summary -------------
    metrics = {
        "dataset": {**report, "n_train": len(X_train), "n_test": len(X_test),
                    "test_size": TEST_SIZE, "random_state": RANDOM_STATE},
        "models": results, "selected_model": selected,
        "selection_criterion": "Highest mean 5-fold cross-validated F2-score on the training set",
        "selection_reason": reason,
        "feature_importance": importance, "lr_coefficients": coefficients,
        "sklearn_version": sklearn.__version__,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for name, r in results.items():
        slug = name.lower().replace(" ", "_")
        plot_confusion_matrix(r["test"]["confusion_matrix"], f"{name} - confusion matrix").savefig(
            FIGURES_DIR / f"confusion_matrix_{slug}.png", dpi=150, bbox_inches="tight")
    plot_model_comparison({n: r["test"] for n, r in results.items()}).savefig(
        FIGURES_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plot_roc_curves({n: r["roc"] for n, r in results.items()}).savefig(
        FIGURES_DIR / "roc_curves.png", dpi=150, bbox_inches="tight")
    plot_feature_importance(importance).savefig(
        FIGURES_DIR / "feature_importance_rf.png", dpi=150, bbox_inches="tight")

    lines = ["| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | False negatives |",
             "|---|---|---|---|---|---|---|"]
    for n, r in results.items():
        t = r["test"]
        lines.append(f"| {n} | {t['accuracy']:.3f} | {t['precision']:.3f} | {t['recall']:.3f} | "
                     f"{t['f1']:.3f} | {t['roc_auc']:.3f} | {t['fn']} |")
    (METRICS_PATH.parent / "results_summary.md").write_text(
        "# Test-set results (auto-generated by src/train_model.py)\n\n" + "\n".join(lines)
        + f"\n\n**Final model:** {selected}\n\n{reason}\n")
    print("Metrics, figures and results_summary.md saved. Training complete.")


if __name__ == "__main__":
    main()
